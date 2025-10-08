/****************************************************************************************
* FileName     : mmw_spi_comm.c
*
* Description  : This file implements mmwave link example application for SPI communication
*				 to mmwave sensor (AWR1243/AWR2243).
*
****************************************************************************************
* (C) Copyright 2020, Texas Instruments Incorporated. - TI web address www.ti.com
*---------------------------------------------------------------------------------------
*
*  Redistribution and use in source and binary forms, with or without modification,
*  are permitted provided that the following conditions are met:
*
*    Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
*    Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
*    Neither the name of Texas Instruments Incorporated nor the names of its
*    contributors may be used to endorse or promote products derived from this
*    software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
*  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
*  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  OWNER OR CONTRIBUTORS
*  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
*  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
*  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
*  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
*  CONTRACT,  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
*  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/
/******************************************************************************
* INCLUDE FILES
******************************************************************************
*/
#include <windows.h>
#include <stdio.h>
#include <share.h>
#include <string.h>
#include <stdlib.h>
#include "mmw_spi_comm.h"
#include <ti/control/mmwavelink/mmwavelink.h>
#include <math.h>
#include "ti\example\platform\mmwl_ftdi\mmwl_port_ftdi.h"
#include "rls_osi.h"
#include "mmw_config.h"

/* AWR2243 meta image file,
suggestion to flash the firmware to AWR12/22xx EVM first. */
#include "firmware\xwr22xx_metaImage.h"

/****************************************************************************************
* MACRO DEFINITIONS
****************************************************************************************
*/
#define MMWL_FW_FIRST_CHUNK_SIZE (220U)
#define MMWL_FW_CHUNK_SIZE (228U)
#define MMWL_META_IMG_FILE_SIZE (sizeof(metaImage))
#define MMWL_APP_PSUEDO_MSG							 (0xFFU)
#define RL_RF_DCBIST_SIG_PSEUDO_REPORT				 (0xFFU)
#define RL_VERSION_PSEUDO_REPORT					 (0xFEU)
#define GET_BIT_VALUE(data, noOfBits, location)    ((((rlUInt32_t)(data)) >> (location)) &\
                                               (((rlUInt32_t)((rlUInt32_t)1U << (noOfBits))) - (rlUInt32_t)1U))
/* Async Event Timeouts */
#define MMWL_API_INIT_TIMEOUT                (2000) /* 2 Sec*/
#define MMWL_API_START_TIMEOUT               (1000) /* 1 Sec*/
#define MMWL_API_RF_INIT_TIMEOUT             (1000) /* 1 Sec*/

/* MAX unique chirp AWR2243 supports */
#define MAX_UNIQUE_CHIRP_INDEX                (512 -1)

/* MAX index to read back chirp config  */
#define MAX_GET_CHIRP_CONFIG_IDX              14

/* LUT Buffer size for Advanced chirp 
   Max size = 12KB (12*1024) */
#define LUT_ADVCHIRP_TABLE_SIZE                5*1024

#define _CAPTURE_TO_FILE_
#define DEBUG_TRACE_EN

#ifndef DEBUG_PRINTF_EN
#define PRINT_FUNC    printf
#else
#define PRINT_FUNC
#endif

extern void mmw_TriggerDcaCapture(void);
extern uint64_t computeCRC(uint8_t *p, uint32_t len, uint8_t width);

/******************************************************************************
* GLOBAL VARIABLES/DATA-TYPES DEFINITIONS
******************************************************************************
*/
typedef int (*RL_P_OS_SPAWN_FUNC_PTR)(RL_P_OSI_SPAWN_ENTRY pEntry, const void* pValue, unsigned int flags);
typedef int (*RL_P_OS_DELAY_FUNC_PTR)(unsigned int delay);

/* Global Variable for Device Status */
static unsigned char mmwl_bInitComp = 0U;
static unsigned char mmwl_bMssBootErrStatus = 0U;
static unsigned char mmwl_bStartComp = 0U;
static unsigned char mmwl_bRfInitComp = 0U;
/* this variable is being used for monitor report capture task */
unsigned char mmwl_bSensorStarted = 0U;
static unsigned char mmwl_bGpadcDataRcv = 0U;
static unsigned char mmwl_bMssCpuFault = 0U;
static unsigned char mmwl_bMssEsmFault = 0U;
/* monitoring report header count */
unsigned int gMonReportHdrCnt = 0U;
rlUInt16_t gCalMonTimeUnit = 0;
unsigned char gAwr2243CrcType = RL_CRC_TYPE_32BIT;

rlUInt16_t lutOffsetInNBytes = 0;

/* Global variable configurations from config file */
extern rlDevGlobalCfg_t rlDevGlobalCfgArgs;
extern mmwave_sensor_config_t  gMmwSensCfg;
MMWAVE_DEVICE_VARIANT    gDeviceType;

/* store frame periodicity */
unsigned int gFramePeriodicity = 0;
/* store frame count */
unsigned int gFrameCount = 0;

/* SPI Communication handle to AWR2243 device*/
rlComIfHdl_t mmwl_devHdl = NULL;

/* structure parameters of two profile confing and cont mode config are same */
rlProfileCfg_t profileCfgArgs[2] = { 0 };

/* strcture to store dynamic chirp configuration */
rlDynChirpCfg_t dynChirpCfgArgs[3] = { 0 };

/* Strcture to store async event config */
rlRfDevCfg_t rfDevCfg = { 0x0 };

/* Structure to store GPADC measurement data sent by device */
rlRecvdGpAdcData_t rcvGpAdcData = {0};

/* calibData is the calibration data sent by the device which needs to store to
   sFlash and will be used for factory calibration or embedded in the application itself */
rlCalibrationData_t calibData = { 0 };

/* File Handle for Calibration Data */
FILE *CalibrationDataPtr = NULL;

/* Advanced Chirp LUT data */
/* Max size of the LUT is 12KB.
   Maximum of 212 bytes per chunk can be present per SPI message. */
/* This array is created to store the LUT RAM values from the user programmed parameters or config file.
   The populated array is sent over SPI to populate the RadarSS LUT RAM.
   This array is also saved into a file "AdvChirpLUTData.txt" for debug purposes */
/* The chirp paramters start address offset should be 4 byte aligned */
rlInt8_t AdvChirpLUTData[LUT_ADVCHIRP_TABLE_SIZE] = { 0 };

/* File Handle for Advanced Chirp LUT Data */
FILE *AdvChirpLUTDataPtr = NULL;
/* Store recieved Async event counts */
rlUInt32_t asyncEvntCnt0[32];
rlUInt32_t asyncEvntCnt1[32];


/* Function to compare dynamically configured chirp data */
int MMWL_chirpParamCompare(rlChirpCfg_t * chirpData);

#define USE_SYSTEM_TIME
static void rlsGetTimeStamp(char *Tsbuffer)
{
#ifdef USE_SYSTEM_TIME
	SYSTEMTIME SystemTime;
	GetLocalTime(&SystemTime);
	sprintf(Tsbuffer, "[%02d:%02d:%02d:%03d]: ", SystemTime.wHour, SystemTime.wMinute, SystemTime.wSecond, SystemTime.wMilliseconds);
#else
	__int64 tickPerSecond;
	__int64 tick;
	__int64 sec;
	__int64 usec;

	/* Get accuracy */
	QueryPerformanceFrequency((LARGE_INTEGER*)&tickPerSecond);

	/* Get tick */
	QueryPerformanceCounter((LARGE_INTEGER*)&tick);
	sec = (__int64)(tick / tickPerSecond);
	usec = (__int64)((tick - (sec * tickPerSecond)) * 1000000.0 / tickPerSecond);
	sprintf(Tsbuffer, "%07lld.%06lld: ", sec, usec);
#endif
}

FILE* rls_traceFp = NULL;
#ifdef DEBUG_TRACE_EN
/* FTDI Library uses this function for debug prints */
void DEBUG_PRINT(char *fmt, ...)
{
	char cBuffer[1000];
	if (TRUE)
	{
		va_list ap;
		va_start(ap, fmt);
		vsnprintf(&cBuffer[0], sizeof(cBuffer), fmt, ap);
#ifdef _CAPTURE_TO_FILE_
		if (rls_traceFp != NULL)
		{
			char tsTime[30] = { 0 };
			rlsGetTimeStamp(tsTime);
			fwrite(tsTime, sizeof(char), strlen(tsTime), rls_traceFp);
			fwrite(cBuffer, sizeof(char), strlen(cBuffer), rls_traceFp);
			fflush(rls_traceFp);
		}
		else
		{
			char tsTime[30] = { 0 };
			rls_traceFp = _fsopen("trace.txt", "wt", _SH_DENYWR);
			rlsGetTimeStamp(tsTime);
			fwrite(tsTime, sizeof(char), strlen(tsTime), rls_traceFp);
			fwrite(cBuffer, sizeof(char), strlen(cBuffer), rls_traceFp);
			fflush(rls_traceFp);
		}
#endif
		va_end(ap);
	}
}
#else
#define DEBUG_PRINT
#endif


/******************************************************************************
* all function definations starts here
*******************************************************************************
*/
/** @fn int MMWL_SwapResetAndPowerOn(rlUInt8_t deviceMap)
*
*   @brief Swap ROM with RAM, resets the core and waits for power-on completion
*   @param[in] deviceMap - device map
*
*   @return int Success - 0, Failure - Error Code 
*	@Note: For AWR2243 ES1.0 only
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_SwapResetAndPowerOn(rlUInt8_t deviceMap)
{
    int timeOutCnt = 0;
    int retVal = RL_RET_CODE_OK;
	
	/* Wait for the below async events only when firmware download over SPI is done.
	   When booting from sFlash, it is not required to wait for below async events */
	if (!rlDevGlobalCfgArgs.DisableFwDownload)
	{
		/* Wait for MSS ESM fault */
		/* Wait for MSS Boot error status if flash is not connected */
		while ((mmwl_bMssEsmFault == 0U) || (mmwl_bMssBootErrStatus == 0U))
		{
			osiSleep(1); /*Sleep 1 msec*/
			timeOutCnt++;
			if (timeOutCnt > MMWL_API_START_TIMEOUT)
			{
				break;
			}
			/* If flash is connected, then no need to wait for MSS boot error status */
			if (rlDevGlobalCfgArgs.IsFlashConnected)
			{
				mmwl_bMssBootErrStatus = 1U;
			}
		}
	}
	mmwl_bMssEsmFault = 0U;
	mmwl_bMssBootErrStatus = 0U;

	/* Disable MSS Watchdog */
	retVal = rlDeviceSetInternalConf(deviceMap, 0xFFFFFF0C, 0x000000AD);

    /* Swap RAM memory map with ROM memory map  */
    retVal = rlDeviceSetInternalConf(deviceMap, 0xFFFFFF20, 0x00ADAD00);

    /* Disable the Ack*/
    rlDeviceConfigureAckTimeout(0);

    /* reset the core */
    retVal = rlDeviceSetInternalConf(deviceMap, 0xFFFFFF04, 0x000000AD);

    /* Enable the Ack*/
    rlDeviceConfigureAckTimeout(1000);

    /* Wait for Power ON complete */
    if (0 == retVal)
    {
        timeOutCnt = 0;
        while (mmwl_bInitComp == 0)
        {
            osiSleep(1); //Sleep 1 msec
            timeOutCnt++;
            if (timeOutCnt > MMWL_API_INIT_TIMEOUT)
            {
                retVal = RL_RET_CODE_RESP_TIMEOUT;
                break;
            }
        }
    }
    mmwl_bInitComp = 0U;
    return retVal;
}

/** @fn void MMWL_asyncEventHandler(rlUInt8_t deviceIndex, rlUInt16_t sbId,
*    rlUInt16_t sbLen, rlUInt8_t *payload)
*
*   @brief Radar Async Event Handler callback
*   @param[in] msgId - Message Id
*   @param[in] sbId - SubBlock Id
*   @param[in] sbLen - SubBlock Length
*   @param[in] payload - Sub Block Payload
*
*   @return None
*
*   Radar Async Event Handler callback
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
void MMWL_asyncEventHandler(rlUInt8_t deviceIndex, rlUInt16_t sbId,
	rlUInt16_t sbLen, rlUInt8_t *payload)
{
	unsigned int deviceMap = 0;
	rlUInt16_t msgId = sbId / RL_MAX_SB_IN_MSG;
	rlUInt16_t asyncSB = RL_GET_SBID_FROM_MSG(sbId, msgId);

	/* Host can receive Async Event from RADARSS/MSS */
	switch (msgId)
	{
		/* Async Event from RADARSS */
		case RL_RF_ASYNC_EVENT_MSG:
		{
			asyncEvntCnt0[asyncSB]++;
			switch (asyncSB)
			{
			case RL_RF_AE_INITCALIBSTATUS_SB:
			{
				mmwl_bRfInitComp = 1U;
				report_write(payload, msgId, asyncSB, gDeviceType);
			}
			break;
			case RL_RF_AE_FRAME_TRIGGER_RDY_SB:
			{
				gFrameTriggerTime = GetTickCount();
				mmwl_bSensorStarted = 1U;
				PRINT_FUNC("Async event: Frame trigger \n");
			}
			break;
			case RL_RF_AE_FRAME_END_SB:
			{
				mmwl_bSensorStarted = 0U;
				PRINT_FUNC("Async event: Frame stopped \n");
			}
			break;
			case RL_RF_AE_MON_TIMING_FAIL_REPORT_SB:
			{
				PRINT_FUNC("Aync event: Monitoring Timing Failed Report [%d] \n",\
					((rlCalMonTimingErrorReportData_t*)payload)->timingFailCode);
				break;
			}
			case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
			{
				// mmwl_bRunTimeCalib = 1U;
				report_write(payload, msgId, asyncSB, gDeviceType);
				break;
			}
			case RL_RF_AE_CPUFAULT_SB:
			{
				unsigned int *lPayload = (unsigned int *)payload;
				PRINT_FUNC("BSS CPU fault [Payload]");
				while (sbLen--)
				{
					PRINT_FUNC("0x%08x ", *(lPayload++));
				}
				PRINT_FUNC("\n");
				exit(RL_RF_AE_CPUFAULT_SB);
				break;
			}
			case RL_RF_AE_ESMFAULT_SB:
			{
				PRINT_FUNC("BSS ESM fault \n");
				break;
			}
			//Digital Latentfault monitor entry to async event handler
			case RL_RF_AE_DIG_LATENTFAULT_REPORT_SB:
			case RL_RF_AE_MON_TEMPERATURE_REPORT_SB:
			case RL_RF_AE_MON_RX_GAIN_PHASE_REPORT:
			case RL_RF_AE_MON_RX_IF_STAGE_REPORT:
			case RL_RF_AE_MON_TX0_POWER_REPORT:
			case RL_RF_AE_MON_TX1_POWER_REPORT:
			case RL_RF_AE_MON_TX2_POWER_REPORT:
			case RL_RF_AE_MON_TX0_BALLBREAK_REPORT:
			case RL_RF_AE_MON_TX1_BALLBREAK_REPORT:
			case RL_RF_AE_MON_RX_NOISE_FIG_REPORT:
			{
				report_write(payload, msgId, asyncSB, gDeviceType);
				break;
			}

			case RL_RF_AE_MON_REPORT_HEADER_SB:
			{
				PRINT_FUNC("Aync event: Monitor Report Header FTTI count [%d] \n", ((rlMonReportHdrData_t*)payload)->fttiCount);
				gMonReportHdrCnt++;
				break;
			}
			default:
			{
				PRINT_FUNC("Unhandled RadarSS Aync Event msgId: 0x%x, asyncSB:0x%x  \n", msgId, asyncSB);
				break;
			}
			}
		}
		break;
		/* async event from radarSS which sub-block IDs comes under 0x81 MsgID */
		case RL_RF_ASYNC_EVENT_1_MSG:
		{
			asyncEvntCnt1[asyncSB]++;
			switch (asyncSB)
			{
			case RL_RF_AE_MON_TX2_BALLBREAK_REPORT:
			case RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT:
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
				report_write(payload, msgId, asyncSB, gDeviceType);
				break;
			}
			default:
			{
				PRINT_FUNC("Unhandled Aync Event msgId: 0x%x, asyncSB:0x%x  \n", msgId, asyncSB);
				break;
			}
			}
			break;
		}
		/* Async Event from MSS */
		case RL_DEV_ASYNC_EVENT_MSG:
		{
			switch (asyncSB)
			{
			case RL_DEV_AE_MSSPOWERUPDONE_SB:
			{
				mmwl_bInitComp = 1U;
			}
			break;
			case RL_DEV_AE_MSS_BOOTERRSTATUS_SB:
			{
				mmwl_bInitComp = 1U;
			}
			break;
			case RL_DEV_AE_RFPOWERUPDONE_SB:
			{
				mmwl_bStartComp = 1U;
			}
			break;
			case RL_DEV_AE_MSS_ESMFAULT_SB:
			{
				PRINT_FUNC("MSS ESM Error \n");
			}
			break;
			case RL_DEV_AE_MSS_CPUFAULT_SB:
			{
				PRINT_FUNC("MSS CPU Fault\n");
				mmwl_bMssCpuFault = 1U;
			}
			break;
			default:
			{
				PRINT_FUNC("Unhandled Aync Event msgId: 0x%x, asyncSB:0x%x  \n", msgId, asyncSB);
				break;
			}
			}
		}
		break;
		default:
		{
			PRINT_FUNC("Unhandled Aync Event via ---msgId---: 0x%x, asyncSB:0x%x  \n", msgId, asyncSB);
			break;
		}

	}
}

/** @fn int MMWL_enableDevice(unsigned char deviceIndex)
*
*   @brief Performs SOP and enables the device.
*
*   @param[in] deviceIndex
*
*   @return int Success - 0, Failure - Error Code
*
*   Power on Slave device API.
*/
int MMWL_enableDevice(unsigned char deviceIndex)
{
    int retVal = RL_RET_CODE_OK;
    /* Enable device in Functional Mode (SOP-4) */
    PRINT_FUNC("rlDeviceEnable Callback is called by mmWaveLink for Device Index [%d]\n\n", deviceIndex);
    return rlsEnableDevice(deviceIndex);
}

/** @fn int MMWL_disableDevice(unsigned char deviceIndex)
*
*   @brief disables the device.
*
*   @param[in] deviceIndex
*
*   @return int Success - 0, Failure - Error Code
*
*   Power on Slave device API.
*/
int MMWL_disableDevice(unsigned char deviceIndex)
{
    PRINT_FUNC("rlDeviceDisable Callback is called by mmWaveLink for Device Index [%d]\n\n", deviceIndex);
    return rlsDisableDevice(deviceIndex);
}

/** @fn int MMWL_computeCRC(unsigned char* data, unsigned int dataLen, unsigned char crcLen,
                        unsigned char* outCrc)
*
*   @brief Compute the CRC of given data
*
*   @param[in] data - message data buffer pointer
*    @param[in] dataLen - length of data buffer
*    @param[in] crcLen - length of crc 2/4/8 bytes
*    @param[out] outCrc - computed CRC data
*
*   @return int Success - 0, Failure - Error Code
*
*   Compute the CRC of given data
*/
int MMWL_computeCRC(unsigned char* data, unsigned int dataLen, unsigned char crcLen,
                        unsigned char* outCrc)
{
    uint64_t crcResult = computeCRC(data, dataLen, (16 << crcLen));
    memcpy(outCrc, &crcResult, (2 << crcLen));
    return 0;
}
extern unsigned long i2cAddr[RLS_NUM_CONNECTED_DEVICES_MAX];
/** @fn int MMWL_powerOnMaster(deviceMap)
*
*   @brief Power on Master API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Power on Master API.
*/
int MMWL_powerOnMaster(unsigned char deviceMap, int deviceType)
{
    int retVal = RL_RET_CODE_OK, timeOutCnt = 0;
    /*
     \subsection     porting_step1   Step 1 - Define mmWaveLink client callback structure
    The mmWaveLink framework is ported to different platforms using mmWaveLink client callbacks. These
    callbacks are grouped as different structures such as OS callbacks, Communication Interface
    callbacks and others. Application needs to define these callbacks and initialize the mmWaveLink
    framework with the structure.

     Refer to \ref rlClientCbs_t for more details
     */
    rlClientCbs_t *clientCtx = &rlDevGlobalCfgArgs.clientCtx;

    /* store CRC Type which has been read from mmwaveconfig.txt file */
    gAwr2243CrcType = clientCtx->crcType;

    /*
    \subsection     porting_step2   Step 2 - Implement Communication Interface Callbacks
    The mmWaveLink device support several standard communication protocol among SPI and MailBox
    Depending on device variant, one need to choose the communication channel. For e.g
    xWR1443/xWR1642 requires Mailbox interface and AWR2243 supports SPI interface.
    The interface for this communication channel should include 4 simple access functions:
    -# rlComIfOpen
    -# rlComIfClose
    -# rlComIfRead
    -# rlComIfWrite

    Refer to \ref rlComIfCbs_t for interface details
    */
	if (rlDevGlobalCfgArgs.TransferMode == 0)
	{
		clientCtx->comIfCb.rlComIfOpen = rlsCommOpen;
		clientCtx->comIfCb.rlComIfClose = rlsCommClose;
		clientCtx->comIfCb.rlComIfRead = rlsSpiRead;
		clientCtx->comIfCb.rlComIfWrite = rlsSpiWrite;
	}
	else
	{
		clientCtx->comIfCb.rlComIfOpen = rlsCommOpen;
		clientCtx->comIfCb.rlComIfClose = rlsCommClose;
		clientCtx->comIfCb.rlComIfRead = rlsI2cRead;
		clientCtx->comIfCb.rlComIfWrite = rlsI2cWrite;
		i2cAddr[0] = 0x28;
	}

    /*   \subsection     porting_step3   Step 3 - Implement Device Control Interface
    The mmWaveLink driver internally powers on/off the mmWave device. The exact implementation of
    these interface is platform dependent, hence you need to implement below functions:
    -# rlDeviceEnable
    -# rlDeviceDisable
    -# rlRegisterInterruptHandler

    Refer to \ref rlDeviceCtrlCbs_t for interface details
    */
    clientCtx->devCtrlCb.rlDeviceDisable = MMWL_disableDevice;
    clientCtx->devCtrlCb.rlDeviceEnable = MMWL_enableDevice;
    clientCtx->devCtrlCb.rlDeviceMaskHostIrq = rlsCommIRQMask;
    clientCtx->devCtrlCb.rlDeviceUnMaskHostIrq = rlsCommIRQUnMask;
    clientCtx->devCtrlCb.rlRegisterInterruptHandler = rlsRegisterInterruptHandler;
    clientCtx->devCtrlCb.rlDeviceWaitIrqStatus = rlsDeviceWaitIrqStatus;

    /*  \subsection     porting_step4     Step 4 - Implement Event Handlers
    The mmWaveLink driver reports asynchronous event indicating mmWave device status, exceptions
    etc. Application can register this callback to receive these notification and take appropriate
    actions

    Refer to \ref rlEventCbs_t for interface details*/
    clientCtx->eventCb.rlAsyncEvent = MMWL_asyncEventHandler;

    /*  \subsection     porting_step5     Step 5 - Implement OS Interface
    The mmWaveLink driver can work in both OS and NonOS environment. If Application prefers to use
    operating system, it needs to implement basic OS routines such as tasks, mutex and Semaphore


    Refer to \ref rlOsiCbs_t for interface details
    */
    /* Mutex */
    clientCtx->osiCb.mutex.rlOsiMutexCreate = osiLockObjCreate;
    clientCtx->osiCb.mutex.rlOsiMutexLock = osiLockObjLock;
    clientCtx->osiCb.mutex.rlOsiMutexUnLock = osiLockObjUnlock;
    clientCtx->osiCb.mutex.rlOsiMutexDelete = osiLockObjDelete;

    /* Semaphore */
    clientCtx->osiCb.sem.rlOsiSemCreate = osiSyncObjCreate;
    clientCtx->osiCb.sem.rlOsiSemWait = osiSyncObjWait;
    clientCtx->osiCb.sem.rlOsiSemSignal = osiSyncObjSignal;
    clientCtx->osiCb.sem.rlOsiSemDelete = osiSyncObjDelete;

    /* Spawn Task */
    clientCtx->osiCb.queue.rlOsiSpawn = (RL_P_OS_SPAWN_FUNC_PTR)osiSpawn;

    /* Sleep/Delay Callback*/
    clientCtx->timerCb.rlDelay = (RL_P_OS_DELAY_FUNC_PTR)osiSleep;

    /*  \subsection     porting_step6     Step 6 - Implement CRC Interface
    The mmWaveLink driver uses CRC for message integrity. If Application prefers to use
    CRC, it needs to implement CRC routine.

    Refer to \ref rlCrcCbs_t for interface details
    */
    clientCtx->crcCb.rlComputeCRC = MMWL_computeCRC;

    /*  \subsection     porting_step7     Step 7 - Define Platform
    The mmWaveLink driver can be configured to run on different platform by
    passing appropriate platform and device type
    */
    clientCtx->platform = RL_PLATFORM_HOST;
	if (deviceType == AWR2243)
	{
		clientCtx->arDevType = RL_AR_DEVICETYPE_22XX;
	}
	else
	{
		clientCtx->arDevType = RL_AR_DEVICETYPE_12XX;
	}

    /*clear all the interupts flag*/
    mmwl_bInitComp = 0;
    mmwl_bStartComp = 0U;
    mmwl_bRfInitComp = 0U;

    /*  \subsection     porting_step8     step 8 - Call Power ON API and pass client context
    The mmWaveLink driver initializes the internal components, creates Mutex/Semaphore,
    initializes buffers, register interrupts, bring mmWave front end out of reset.
    */
    retVal = rlDevicePowerOn(deviceMap, *clientCtx);

    /*  \subsection     porting_step9     step 9 - Test if porting is successful
    Once configuration is complete and mmWave device is powered On, mmWaveLink driver receives
    asynchronous event from mmWave device and notifies application using
    asynchronous event callback.
    Note: In case of AWR2243 ES1.0, Host needs to wait for MSS CPU Fault as well, with current 
    ROM version this MSS CPU fault async-event sent by AWR device which Host needs to ignore.

    Refer to \ref MMWL_asyncEventHandler for event details
    */
    while ((mmwl_bInitComp == 0U) || (mmwl_bMssCpuFault == 0))
    {
        osiSleep(1); /*Sleep 1 msec*/
        timeOutCnt++;
        if (timeOutCnt > MMWL_API_INIT_TIMEOUT)
        {
            retVal = RL_RET_CODE_RESP_TIMEOUT;
            break;
        }
    }
    mmwl_bMssCpuFault = 0U;
    mmwl_bInitComp = 0U;
    return retVal;
}

int MMWL_fileWrite(unsigned char deviceMap,
                unsigned short remChunks,
                unsigned short chunkLen,
                unsigned char *chunk)
{
    int ret_val = -1;

    rlFileData_t fileChunk = { 0 };
    fileChunk.chunkLen = chunkLen;
    memcpy(fileChunk.fData, chunk, chunkLen);

    ret_val = rlDeviceFileDownload(deviceMap, &fileChunk, remChunks);
    return ret_val;
}

/** @fn int MMWL_fileDownload((unsigned char deviceMap,
                  mmwlFileType_t fileType,
                  unsigned int fileLen)
*
*   @brief Firmware Download API.
*
*   @param[in] deviceMap - Devic Index
*    @param[in] fileType - firmware/file type
*    @param[in] fileLen - firmware/file length
*
*   @return int Success - 0, Failure - Error Code
*
*   Firmware Download API.
*/
int MMWL_fileDownload(unsigned char deviceMap,
                  unsigned int fileLen)
{
    unsigned int imgLen = fileLen;
    int ret_val = -1;
    int mmwl_iRemChunks = 0;
    unsigned short usChunkLen = 0U;
    unsigned int iNumChunks = 0U;
    unsigned short usLastChunkLen = 0;
    unsigned short usFirstChunkLen = 0;
    unsigned short usProgress = 0;

    /*First Chunk*/
    unsigned char firstChunk[MMWL_FW_CHUNK_SIZE];
    unsigned char* pmmwl_imgBuffer = NULL;

    pmmwl_imgBuffer = (unsigned char*)&metaImage[0];

    if(pmmwl_imgBuffer == NULL)
    {
        PRINT_FUNC("MMWL_fileDwld Fail : File Buffer is NULL \n\n\r");
        return -1;
    }

    /*Download to Device*/
    usChunkLen = MMWL_FW_CHUNK_SIZE;
    iNumChunks = (imgLen + 8) / usChunkLen;
    mmwl_iRemChunks = iNumChunks;

    if (mmwl_iRemChunks > 0)
    {
        usLastChunkLen = (imgLen + 8) % usChunkLen;
        usFirstChunkLen = MMWL_FW_CHUNK_SIZE;
		mmwl_iRemChunks += 1;
    }
    else
    {
        usFirstChunkLen = imgLen + 8;
    }

    *((unsigned int*)&firstChunk[0]) = (unsigned int)MMWL_FILETYPE_META_IMG;
    *((unsigned int*)&firstChunk[4]) = (unsigned int)imgLen;
    memcpy((char*)&firstChunk[8], (char*)pmmwl_imgBuffer,
                usFirstChunkLen - 8);

    ret_val = MMWL_fileWrite(deviceMap, (mmwl_iRemChunks-1), usFirstChunkLen,
                              firstChunk);
    if (ret_val < 0)
    {
        PRINT_FUNC("MMWL_fileDwld Fail : Ftype: %d\n\n\r", MMWL_FILETYPE_META_IMG);
        return ret_val;
    }
    pmmwl_imgBuffer += MMWL_FW_FIRST_CHUNK_SIZE;
    mmwl_iRemChunks--;

    if(mmwl_iRemChunks > 0)
    {
        printf("Download in Progress: %d%%..", usProgress);
    }
    /*Remaining Chunk*/
    while (mmwl_iRemChunks > 0)
    {
        if ((((iNumChunks - mmwl_iRemChunks) * 100)/iNumChunks - usProgress) > 10)
        {
            usProgress += 10;
			printf("%d%%..", usProgress);
        }

		/* Last chunk */
		if ((mmwl_iRemChunks == 1) && (usLastChunkLen > 0))
		{
			ret_val = MMWL_fileWrite(deviceMap, 0, usLastChunkLen,
				pmmwl_imgBuffer);
			if (ret_val < 0)
			{
				PRINT_FUNC("MMWL_fileDwld last chunk Fail : Ftype: %d\n\n\r",
					MMWL_FILETYPE_META_IMG);
				return ret_val;
			}
		}
		else
		{
			ret_val = MMWL_fileWrite(deviceMap, (mmwl_iRemChunks - 1),
				MMWL_FW_CHUNK_SIZE, pmmwl_imgBuffer);

			if (ret_val < 0)
			{
				PRINT_FUNC("\n\n\r MMWL_fileDwld rem chunk Fail : Ftype: %d\n\n\r",
					MMWL_FILETYPE_META_IMG);
				return ret_val;
			}
			pmmwl_imgBuffer += MMWL_FW_CHUNK_SIZE;
		}

        mmwl_iRemChunks--;
    }
	printf("Done!\n\n");
    return ret_val;
}

/** @fn int MMWL_firmwareDownload(deviceMap)
*
*   @brief Firmware Download API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Firmware Download API.
*/
int MMWL_firmwareDownload(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK, timeOutCnt = 0;

    /* Meta Image download */
    PRINT_FUNC("Meta Image download started for deviceMap %u\n\n",
        deviceMap);
    retVal = MMWL_fileDownload(deviceMap, MMWL_META_IMG_FILE_SIZE);
    PRINT_FUNC("Meta Image download complete ret = %d\n\n", retVal);

    return retVal;
}

/** @fn int MMWL_rfEnable(deviceMap)
*
*   @brief RFenable API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   RFenable API.
*/
int MMWL_rfEnable(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK, timeOutCnt = 0;
    retVal = rlDeviceRfStart(deviceMap);
    while (mmwl_bStartComp == 0U)
    {
        osiSleep(1); /*Sleep 1 msec*/
        timeOutCnt++;
        if (timeOutCnt > MMWL_API_START_TIMEOUT)
        {
            retVal = RL_RET_CODE_RESP_TIMEOUT;
            break;
        }
    }
    mmwl_bStartComp = 0;

    if(retVal == RL_RET_CODE_OK)
    {
        rlVersion_t verArgs = {0};
		rlRfDieIdCfg_t dieId = { 0 };
        retVal = rlDeviceGetVersion(deviceMap,&verArgs);
		if(rlDevGlobalCfgArgs.monitorCaptureEn == 1)
			report_write((unsigned char*)&verArgs, MMWL_APP_PSUEDO_MSG, \
						  RL_VERSION_PSEUDO_REPORT, gDeviceType);

        PRINT_FUNC("RF Version [%2d.%2d.%2d.%2d] \nMSS version [%2d.%2d.%2d.%2d] \nmmWaveLink version [%2d.%2d.%2d.%2d]\n\n",
            verArgs.rf.fwMajor, verArgs.rf.fwMinor, verArgs.rf.fwBuild, verArgs.rf.fwDebug,
            verArgs.master.fwMajor, verArgs.master.fwMinor, verArgs.master.fwBuild, verArgs.master.fwDebug,
            verArgs.mmWaveLink.major, verArgs.mmWaveLink.minor, verArgs.mmWaveLink.build, verArgs.mmWaveLink.debug);
        PRINT_FUNC("RF Patch Version [%2d.%2d.%2d.%2d] \nMSS Patch version [%2d.%2d.%2d.%2d]\n\n",
            verArgs.rf.patchMajor, verArgs.rf.patchMinor, ((verArgs.rf.patchBuildDebug & 0xF0) >> 4), (verArgs.rf.patchBuildDebug & 0x0F),
            verArgs.master.patchMajor, verArgs.master.patchMinor, ((verArgs.master.patchBuildDebug & 0xF0) >> 4), (verArgs.master.patchBuildDebug & 0x0F));

		retVal = rlGetRfDieId(deviceMap, &dieId);

		PRINT_FUNC("Lot Number [%d] \nWafer Number [%d] \nDie Coordinates in Wafer ([%d], [%d]) \n\n", dieId.lotNo, dieId.waferNo, dieId.devX, dieId.devY);

    }
    return retVal;
}

/** @fn int MMWL_dataFmtConfig(unsigned char deviceMap)
*
*   @brief Data Format Config API
*
*   @return Success - 0, Failure - Error Code
*
*   Data Format Config API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_dataFmtConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDevDataFmtCfg_t dataFmtCfgArgs = gMmwSensCfg.datapathCfg.devDataFmtCfg;

    retVal = rlDeviceSetDataFmtConfig(deviceMap, &dataFmtCfgArgs);
    return retVal;
}

/** @fn int MMWL_ldoBypassConfig(unsigned char deviceMap)
*
*   @brief LDO Bypass Config API
*
*   @return Success - 0, Failure - Error Code
*
*   LDO Bypass Config API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_ldoBypassConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlRfLdoBypassCfg_t rfLdoBypassCfgArgs = { 0 };

    PRINT_FUNC("Calling rlRfSetLdoBypassConfig With Bypass [%d] \n\n",
        rfLdoBypassCfgArgs.ldoBypassEnable);

    retVal = rlRfSetLdoBypassConfig(deviceMap, &rfLdoBypassCfgArgs);
    return retVal;
}

/** @fn int MMWL_adcOutConfig(unsigned char deviceMap)
*
*   @brief ADC Configuration API
*
*   @return Success - 0, Failure - Error Code
*
*   ADC Configuration API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_adcOutConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;

    rlAdcOutCfg_t adcOutCfgArgs = gMmwSensCfg.rfCfg.adcOutCfg;

    PRINT_FUNC("Calling rlSetAdcOutConfig With [%d]ADC Bits and [%d]ADC Format \n\n",
        adcOutCfgArgs.fmt.b2AdcBits, adcOutCfgArgs.fmt.b2AdcOutFmt);

    retVal = rlSetAdcOutConfig(deviceMap, &adcOutCfgArgs);
    return retVal;
}

/** @fn int MMWL_channelConfig(unsigned char deviceMap,
                               unsigned short cascading)
*
*   @brief Channel Config API
*
*   @return Success - 0, Failure - Error Code
*
*   Channel Config API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_channelConfig(unsigned char deviceMap,
                       unsigned short cascade)
{
    int retVal = RL_RET_CODE_OK;
    /* TBD - Read GUI Values */
	rlChanCfg_t rfChanCfgArgs = gMmwSensCfg.rfCfg.channelCfg;

    PRINT_FUNC("Calling rlSetChannelConfig With [%d]Rx and [%d]Tx Channel Enabled \n\n",
           rfChanCfgArgs.rxChannelEn, rfChanCfgArgs.txChannelEn);

    retVal = rlSetChannelConfig(deviceMap, &rfChanCfgArgs);
    return retVal;
}

/** @fn int MMWL_setAsyncEventDir(unsigned char deviceMap)
*
*   @brief Update async event message direction and CRC type of Async event
*           from AWR2243 radarSS to Host
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Update async event message direction and CRC type of Async event
*   from AWR2243 radarSS to Host
*/
int MMWL_setAsyncEventDir(unsigned char  deviceMap)
{
    int32_t         retVal;
    /* set global and monitoring async event direction to Host */
    rfDevCfg.aeDirection = 0x05;
    /* Set the CRC type of Async event received from radarSS */
    rfDevCfg.aeCrcConfig = gAwr2243CrcType;
    retVal = rlRfSetDeviceCfg(deviceMap, &rfDevCfg);
    return retVal;
}

/** @fn int MMWL_setMiscConfig(unsigned char deviceMap)
*
*   @brief Sets misc feature such as per chirp phase shifter and Advance chirp
*
*   @param[in] deviceMap - Device Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Sets misc feature such as per chirp phase shifter and Advance chirp
*/
int MMWL_setMiscConfig(unsigned char deviceMap)
{
	int32_t         retVal;
	rlRfMiscConf_t MiscCfg = { 0 };
	/* Enable Adv chirp feature 
		b0 PERCHIRP_PHASESHIFTER_EN
		b1 ADVANCE_CHIRP_CONFIG_EN  */
	MiscCfg.miscCtl = 0x3;
	retVal = rlRfSetMiscConfig(deviceMap, &MiscCfg);
	return retVal;
}

/** @fn int MMWL_setDeviceCrcType(unsigned char deviceMap)
*
*   @brief Set CRC type of async event from AWR2243 MasterSS
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Set CRC type of async event from AWR2243 MasterSS
*/
int MMWL_setDeviceCrcType(unsigned char deviceMap)
{
    int32_t         retVal;
    rlDevMiscCfg_t devMiscCfg = {0};
    /* Set the CRC Type for Async Event from MSS */
    devMiscCfg.aeCrcConfig = gAwr2243CrcType;
    retVal = rlDeviceSetMiscConfig(deviceMap, &devMiscCfg);
    return retVal;
}

/** @fn int MMWL_basicConfiguration(unsigned char deviceMap, unsigned int cascade)
*
*   @brief Channel, ADC,Data format configuration API.
*
*   @param[in] deviceMap - Devic Index
*    @param[in] unsigned int cascade
*
*   @return int Success - 0, Failure - Error Code
*
*   Channel, ADC,Data format configuration API.
*/
int MMWL_basicConfiguration(unsigned char deviceMap, unsigned int cascade)
{
    int retVal = RL_RET_CODE_OK;

    /* Set which Rx and Tx channels will be enable of the device */
    retVal = MMWL_channelConfig(deviceMap, cascade);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Channel Config failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("Channel Configuration success for deviceMap %u\n\n", deviceMap);
    }
    /* ADC out data format configuration */
    retVal = MMWL_adcOutConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("AdcOut Config failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("AdcOut Configuration success for deviceMap %u\n\n", deviceMap);
    }

    /* LDO bypass configuration */
    retVal = MMWL_ldoBypassConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("LDO Bypass Config failed for deviceMap %u with error code %d\n\n",
            deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("LDO Bypass Configuration success for deviceMap %u\n\n", deviceMap);
    }

    /* Data format configuration */
    retVal = MMWL_dataFmtConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Data format Configuration failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("Data format Configuration success for deviceMap %u\n\n", deviceMap);
    }

    /* low power configuration */
    retVal = MMWL_lowPowerConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Low Power Configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("Low Power Configuration success for deviceMap %u \n\n", deviceMap);
    }

    /* Async event direction and control configuration for RadarSS */
    retVal = MMWL_setAsyncEventDir(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("AsyncEvent Configuration failed for deviceMap %u with error code %d \n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("AsyncEvent Configuration success for deviceMap %u \n\n", deviceMap);
    }

	if (rlDevGlobalCfgArgs.LinkAdvChirpTest == TRUE)
	{	/* Misc control configuration for RadarSS */
		/* This API enables the Advanced chirp and per chirp phase shifter features */
		retVal = MMWL_setMiscConfig(deviceMap);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("Misc control configuration failed for deviceMap %u with error code %d \n\n",
				deviceMap, retVal);
			return -1;
		}
		else
		{
			PRINT_FUNC("Misc control configuration success for deviceMap %u \n\n", deviceMap);
		}
	}
	
    return retVal;
}

/** @fn int MMWL_rfInit(unsigned char deviceMap)
*
*   @brief RFinit API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   RFinit API.
*/
int MMWL_rfInit(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK, timeOutCnt = 0;
	rlRfInitCalConf_t rfCalibCfgArgs = { 0 };

    mmwl_bRfInitComp = 0;

	if(memcmp(&gMmwSensCfg.rfCfg.rfInitCalCfg, &rfCalibCfgArgs, sizeof(rlRfInitCalConf_t)) != 0)
	{
		/* RF Init Calibration Configuration */
		retVal = rlRfInitCalibConfig(deviceMap, &gMmwSensCfg.rfCfg.rfInitCalCfg);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("RF Init Calibration Configuration failed for deviceMap %u with error %d \n\n",
				deviceMap, retVal);
			return -1;
		}
		else
		{
			PRINT_FUNC("RF Init Calibration Configuration success for deviceMap %u \n\n", deviceMap);
		}
	}

    retVal = rlRfInit(deviceMap);
    while (mmwl_bRfInitComp == 0U)
    {
        osiSleep(1); /*Sleep 1 msec*/
        timeOutCnt++;
        if (timeOutCnt > MMWL_API_RF_INIT_TIMEOUT)
        {
            retVal = RL_RET_CODE_RESP_TIMEOUT;
            break;
        }
    }
    mmwl_bRfInitComp = 0;
    return retVal;
}

/** @fn int MMWL_saveCalibDataToFile(unsigned char deviceMap)
*
*   @brief Save Calibration Data to a file.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Save Calibration Data to a file.
*/
int MMWL_saveCalibDataToFile(unsigned char deviceMap)
{
	int retVal = RL_RET_CODE_OK;
	int i,j;
	int index = 0;
	char CalibdataBuff[2500] = { 0 };
	CalibrationDataPtr = _fsopen("CalibrationData.txt", "wt", _SH_DENYWR);

	/* Copy data from all the 3 chunks */
	for (i = 0; i < 3; i++)
	{
		sprintf(CalibdataBuff + strlen(CalibdataBuff), "0x%04x\n", calibData.calibChunk[i].numOfChunk);
		sprintf(CalibdataBuff + strlen(CalibdataBuff), "0x%04x\n", calibData.calibChunk[i].chunkId);
		/* Store 224 bytes of data in each chunk in terms of 2 bytes per line */
		for (j = 0; j < 224; j+=2)
		{
			sprintf(CalibdataBuff + strlen(CalibdataBuff), "0x%02x%02x\n", calibData.calibChunk[i].calData[j+1], calibData.calibChunk[i].calData[j]);
		}
	}

	fwrite(CalibdataBuff, sizeof(char), strlen(CalibdataBuff), CalibrationDataPtr);
	fflush(CalibrationDataPtr);

	if (CalibrationDataPtr != NULL)
		fclose(CalibrationDataPtr);

	return retVal;
}

/** @fn int MMWL_LoadCalibDataFromFile(unsigned char deviceMap)
*
*   @brief Load Calibration Data from a file.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Load Calibration Data from a file.
*/
int MMWL_LoadCalibDataFromFile(unsigned char deviceMap)
{
	int retVal = RL_RET_CODE_OK;
	int index = 0;
	char CalibdataBuff[2500] = { 0 };
	char *s, buff[8], val[100];
	int i = 0;
	char readNumChunks = 0, readChunkId = 0;
	
	CalibrationDataPtr = _fsopen("CalibrationData.txt", "rt", _SH_DENYRD);

	if (CalibrationDataPtr == NULL)
	{
		PRINT_FUNC("CalibrationData.txt does not exist or Error opening the file\n\n");
		return -1;
	}

	/*seek the pointer to starting of the file */
	fseek(CalibrationDataPtr, 0, SEEK_SET);

	/*parse the parameters by reading each line of the calib data file*/
	while ((readNumChunks != 3) && (readChunkId != 3))
	{
		unsigned char readDataChunks = 0;
		if ((s = fgets(buff, sizeof buff, CalibrationDataPtr)) != NULL)
		{
			/* Parse value from line */
			s = strtok(buff, "\n");
			if (s == NULL)
			{
				continue;
			}
			else
			{
				strncpy(val, s, STRINGLEN);
				calibData.calibChunk[i].numOfChunk = (rlUInt16_t)strtol(val, NULL, 0);
				readNumChunks++;
			}
		}
		if ((s = fgets(buff, sizeof buff, CalibrationDataPtr)) != NULL)
		{
			/* Parse value from line */
			s = strtok(buff, "\n");
			if (s == NULL)
			{
				continue;
			}
			else
			{
				strncpy(val, s, STRINGLEN);
				calibData.calibChunk[i].chunkId = (rlUInt16_t)strtol(val, NULL, 0);
				readChunkId++;
			}
		}
		while (((s = fgets(buff, sizeof buff, CalibrationDataPtr)) != NULL) && (readDataChunks != 222))
		{
			/* Parse value from line */
			const char* temp = &buff[0];
			char byte1[3];
			char byte2[3];

			strncpy(byte1, temp +4, 2);
			byte1[2] = '\0';
			if (byte1 == NULL)
			{
				continue;
			}
			else
			{
				calibData.calibChunk[i].calData[readDataChunks] = (rlUInt8_t)strtol(byte1, NULL, 16);
				readDataChunks++;
			}

			strncpy(byte2, temp + 2, 2);
			byte2[2] = '\0';
			if (byte2 == NULL)
			{
				continue;
			}
			else
			{
				calibData.calibChunk[i].calData[readDataChunks] = (rlUInt8_t)strtol(byte2, NULL, 16);
				readDataChunks++;
			}
		}
		i++;
	}

	fflush(CalibrationDataPtr);

	if (CalibrationDataPtr != NULL)
		fclose(CalibrationDataPtr);

	return retVal;
}

/** @fn int MMWL_progFiltConfig(unsigned char deviceMap)
*
*   @brief Programmable filter configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Programmable filter configuration API.
*/
int MMWL_progFiltConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
	rlRfProgFiltConf_t progFiltCnfgArgs = { 0 };
	rlRfProgFiltCoeff_t progFiltCoeffCnfgArgs = { 0 };
	/* if this parameter is being set in JSON input file */
	if (memcmp(&progFiltCnfgArgs, &gMmwSensCfg.rfCfg.progFiltCfg, sizeof(rlRfProgFiltConf_t) != 0))
	{
		PRINT_FUNC("Calling rlRfSetProgFiltConfig with \ncoeffStartIdx[%d]\nprogFiltLen[%d] GHz\nprogFiltFreqShift[%d] MHz/uS \n\n",
			progFiltCnfgArgs.coeffStartIdx, progFiltCnfgArgs.progFiltLen, progFiltCnfgArgs.progFiltFreqShift);
		retVal = rlRfSetProgFiltConfig(deviceMap, &gMmwSensCfg.rfCfg.progFiltCfg);

		progFiltCoeffCnfgArgs.coeffArray[0] = -876,
			progFiltCoeffCnfgArgs.coeffArray[1] = -272,
			progFiltCoeffCnfgArgs.coeffArray[2] = 1826,
			progFiltCoeffCnfgArgs.coeffArray[3] = -395,
			progFiltCoeffCnfgArgs.coeffArray[4] = -3672,
			progFiltCoeffCnfgArgs.coeffArray[5] = 3336,
			progFiltCoeffCnfgArgs.coeffArray[6] = 15976,
			progFiltCoeffCnfgArgs.coeffArray[7] = 15976,
			progFiltCoeffCnfgArgs.coeffArray[8] = 3336,
			progFiltCoeffCnfgArgs.coeffArray[9] = -3672,
			progFiltCoeffCnfgArgs.coeffArray[10] = -395,
			progFiltCoeffCnfgArgs.coeffArray[11] = 1826,
			progFiltCoeffCnfgArgs.coeffArray[12] = -272,
			progFiltCoeffCnfgArgs.coeffArray[13] = -876,

			PRINT_FUNC("Calling rlRfSetProgFiltCoeffRam with \ncoeffArray0[%d]\ncoeffArray1[%d] GHz\ncoeffArray2[%d] MHz/uS \n\n",
				progFiltCoeffCnfgArgs.coeffArray[0], progFiltCoeffCnfgArgs.coeffArray[1], progFiltCoeffCnfgArgs.coeffArray[2]);
		retVal = rlRfSetProgFiltCoeffRam(deviceMap, &progFiltCoeffCnfgArgs);

		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("Programmable Filter Configuration failed for deviceMap %u with error code %d \n\n",
				deviceMap, retVal);
		}
		else
		{
			PRINT_FUNC("Programmable Filter Configuration success for deviceMap %u \n\n", deviceMap);
		}
	}
    return retVal;
}

/** @fn int MMWL_profileConfig(unsigned char deviceMap)
*
*   @brief Profile configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Profile configuration API.
*/
int MMWL_profileConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
	
	/* configuring 4 profiles at a time */
	retVal = rlSetProfileConfig(deviceMap, gMmwSensCfg.rfCfg.profileCfgCnt, gMmwSensCfg.rfCfg.profileCfg);

    return retVal;
}

/** @fn int MMWL_chirpConfig(unsigned char deviceMap)
*
*   @brief Chirp configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Chirp configuration API.
*/
int MMWL_chirpConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlChirpCfg_t getChirpCfgArgs[MAX_GET_CHIRP_CONFIG_IDX+1] = {0};

    /* With this API we can configure max 512 chirp in one call */
    retVal = rlSetChirpConfig(deviceMap, gMmwSensCfg.rfCfg.chirpCfgCnt, gMmwSensCfg.rfCfg.chirpCfg);

    return retVal;
}

int MMWL_chirpParamCompare(rlChirpCfg_t * chirpData)
{
    int retVal = RL_RET_CODE_OK, i = 0,j = 0;
    /* compare each chirpConfig parameters to lastly configured via rlDynChirpConfig API */
    while (i <= MAX_GET_CHIRP_CONFIG_IDX)
    {
        if (dynChirpCfgArgs[0].chirpRowSelect == 0x00)
        {
            if ((chirpData->profileId != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 4, 0)) || \
                (chirpData->freqSlopeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 6, 8)) || \
                (chirpData->txEnable != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 3, 16)) || \
                (chirpData->startFreqVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 23, 0)) || \
                (chirpData->idleTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 12, 0)) || \
                (chirpData->adcStartTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 12, 16)))
            {
                break;
            }
            i++;
            chirpData++;
        }
        else if (dynChirpCfgArgs[0].chirpRowSelect == 0x10)
        {
            if ((chirpData->profileId != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 4, 0)) || \
                (chirpData->freqSlopeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 6, 8)) || \
                (chirpData->txEnable != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 3, 16)))
            {
                break;
            }
            i++;
            chirpData++;
            if ((chirpData->profileId != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 4, 0)) || \
                (chirpData->freqSlopeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 6, 8)) || \
                (chirpData->txEnable != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 3, 16)))
            {
                break;
            }
            i++;
            chirpData++;
            if ((chirpData->profileId != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 4, 0)) || \
                (chirpData->freqSlopeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 6, 8)) || \
                (chirpData->txEnable != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 3, 16)))
            {
                break;
            }
            i++;
            chirpData++;
        }
        else if (dynChirpCfgArgs[0].chirpRowSelect == 0x20)
        {
            if (chirpData->startFreqVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 23, 0))
            {
                break;
            }
            i++;
            chirpData++;
            if (chirpData->startFreqVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 23, 0))
            {
                break;
            }
            i++;
            chirpData++;
            if (chirpData->startFreqVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 23, 0))
            {
                break;
            }
            i++;
            chirpData++;
        }
        else if (dynChirpCfgArgs[0].chirpRowSelect == 0x30)
        {
            if ((chirpData->idleTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 12, 0)) || \
                (chirpData->adcStartTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR1, 12, 16)))
            {
                break;
            }
            i++;
            chirpData++;
            if ((chirpData->idleTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 12, 0)) || \
                (chirpData->adcStartTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR2, 12, 16)))
            {
                break;
            }
            i++;
            chirpData++;
            if ((chirpData->idleTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 12, 0)) || \
                (chirpData->adcStartTimeVar != GET_BIT_VALUE(dynChirpCfgArgs[0].chirpRow[j].chirpNR3, 12, 16)))
            {
                break;
            }
            i++;
            chirpData++;
        }
        j++;
    }
    if (i <= MAX_GET_CHIRP_CONFIG_IDX)
    {
        retVal = -1;
    }
    return retVal;
}
/* This function is not Used in this application */
int MMWL_getDynChirpConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK,i = 0, j= 0, chirpNotMatch = 0;
    unsigned short chirpStartIdx;
    rlChirpCfg_t chirpCfgArgs[MAX_GET_CHIRP_CONFIG_IDX+1] = {0};
    if (dynChirpCfgArgs[0].chirpRowSelect == 0x00)
    {
        chirpStartIdx = (dynChirpCfgArgs[0].chirpSegSel * 16);
    }
    else
    {
        chirpStartIdx = (dynChirpCfgArgs[0].chirpSegSel * 48);
    }
    /* get the chirp config for (10+1) chirps for which it's being updated by dynChirpConfig API
       @Note - This examples read back (10+1) num of chirp config for demonstration,
               which user can raise to match with their requirement */
    retVal = rlGetChirpConfig(deviceMap, chirpStartIdx, chirpStartIdx + MAX_GET_CHIRP_CONFIG_IDX, &chirpCfgArgs[0]);

    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("*** Failed - rlGetChirpConfig failed with %d*** \n\n",retVal);
    }

    retVal = MMWL_chirpParamCompare(&chirpCfgArgs[0]);

    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("*** Failed - Parameters are mismatched GetChirpConfig compare to dynChirpConfig *** \n\n");
    }
    else
    {
        PRINT_FUNC("Get chirp configurations are matching with parameters configured via dynChirpConfig \n\n");
    }

    return retVal;
}

/** @fn int MMWL_frameConfig(unsigned char deviceMap)
*
*   @brief Frame configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Frame configuration API.
*/
int MMWL_frameConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlFrameCfg_t frameCfgArgs = gMmwSensCfg.rfCfg.frameCfg;

    gFramePeriodicity = (frameCfgArgs.framePeriodicity * 5)/(1000*1000);
    gFrameCount = frameCfgArgs.numFrames;

    PRINT_FUNC("Calling rlSetFrameConfig with Start Idx[%d] End Idx[%d] Loops[%d] Periodicity[%d]ms \n",
        frameCfgArgs.chirpStartIdx, frameCfgArgs.chirpEndIdx,
        frameCfgArgs.numLoops, (frameCfgArgs.framePeriodicity * 5)/(1000*1000));

    retVal = rlSetFrameConfig(deviceMap, &frameCfgArgs);
    return retVal;
}

/** @fn int MMWL_advFrameConfig(unsigned char deviceMap)
*
*   @brief Advance Frame configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Frame configuration API.
*/
int MMWL_advFrameConfig(unsigned char deviceMap)
{
    int i, retVal = RL_RET_CODE_OK;
    rlAdvFrameCfg_t AdvframeCfgArgs = gMmwSensCfg.rfCfg.advFrameCfg;
    rlAdvFrameCfg_t GetAdvFrameCfgArgs = { 0 };
    /* reset frame periodicity to zero */
    gFramePeriodicity = 0;

    /* Add all subframes periodicity to get whole frame periodicity */
    for (i=0; i < AdvframeCfgArgs.frameSeq.numOfSubFrames; i++)
		gFramePeriodicity += AdvframeCfgArgs.frameSeq.subFrameCfg[i].subFramePeriodicity;

	gFramePeriodicity = (gFramePeriodicity * 5)/(1000*1000);
    /* store total number of frames configured */
    gFrameCount = AdvframeCfgArgs.frameSeq.numFrames;

    PRINT_FUNC("Calling rlSetAdvFrameConfig with numOfSubFrames[%d] forceProfile[%d] numFrames[%d] triggerSelect[%d]ms \n\n",
        AdvframeCfgArgs.frameSeq.numOfSubFrames, AdvframeCfgArgs.frameSeq.forceProfile,
        AdvframeCfgArgs.frameSeq.numFrames, AdvframeCfgArgs.frameSeq.triggerSelect);

    retVal = rlSetAdvFrameConfig(deviceMap, &AdvframeCfgArgs);
    if (retVal == 0)
    {
        retVal = rlGetAdvFrameConfig(deviceMap, &GetAdvFrameCfgArgs);
        if ((AdvframeCfgArgs.frameSeq.forceProfile != GetAdvFrameCfgArgs.frameSeq.forceProfile) || \
            (AdvframeCfgArgs.frameSeq.frameTrigDelay != GetAdvFrameCfgArgs.frameSeq.frameTrigDelay) || \
            (AdvframeCfgArgs.frameSeq.numFrames != GetAdvFrameCfgArgs.frameSeq.numFrames) || \
            (AdvframeCfgArgs.frameSeq.numOfSubFrames != GetAdvFrameCfgArgs.frameSeq.numOfSubFrames) || \
            (AdvframeCfgArgs.frameSeq.triggerSelect != GetAdvFrameCfgArgs.frameSeq.triggerSelect))
        {
            PRINT_FUNC("MMWL_readAdvFrameConfig failed...\n\n");
            return retVal;
        }
    }
    return retVal;
}

/**
 *******************************************************************************
 *
 * \brief   Local function to enable the dummy input of objects from AWR143
 *
 * \param   None
 * /return  retVal   BSP_SOK if the test source is set correctly.
 *
 *******************************************************************************
*/
#if defined (ENABLE_TEST_SOURCE)
int MMWL_testSourceConfig(unsigned char deviceMap)
{
    rlTestSource_t tsArgs = {0};
    rlTestSourceEnable_t tsEnableArgs = {0};
    int retVal = RL_RET_CODE_OK;

    tsArgs.testObj[0].posX = 0;

    tsArgs.testObj[0].posY = 500;
    tsArgs.testObj[0].posZ = 0;

    tsArgs.testObj[0].velX = 0;
    tsArgs.testObj[0].velY = 0;
    tsArgs.testObj[0].velZ = 0;

    tsArgs.testObj[0].posXMin = -32700;
    tsArgs.testObj[0].posYMin = 0;
    tsArgs.testObj[0].posZMin = -32700;

    tsArgs.testObj[0].posXMax = 32700;
    tsArgs.testObj[0].posYMax = 32700;
    tsArgs.testObj[0].posZMax = 32700;

    tsArgs.testObj[0].sigLvl = 150;

    tsArgs.testObj[1].posX = 0;
    tsArgs.testObj[1].posY = 32700;
    tsArgs.testObj[1].posZ = 0;

    tsArgs.testObj[1].velX = 0;
    tsArgs.testObj[1].velY = 0;
    tsArgs.testObj[1].velZ = 0;

    tsArgs.testObj[1].posXMin = -32700;
    tsArgs.testObj[1].posYMin = 0;
    tsArgs.testObj[1].posZMin = -32700;

    tsArgs.testObj[1].posXMax = 32700;
    tsArgs.testObj[1].posYMax = 32700;
    tsArgs.testObj[1].posZMax = 32700;

    tsArgs.testObj[1].sigLvl = 948;

    tsArgs.rxAntPos[0].antPosX = 0;
    tsArgs.rxAntPos[0].antPosZ = 0;
    tsArgs.rxAntPos[1].antPosX = 32;
    tsArgs.rxAntPos[1].antPosZ = 0;
    tsArgs.rxAntPos[2].antPosX = 64;
    tsArgs.rxAntPos[2].antPosZ = 0;
    tsArgs.rxAntPos[3].antPosX = 96;
    tsArgs.rxAntPos[3].antPosZ = 0;

    tsArgs.txAntPos[0].antPosX = 0;
    tsArgs.txAntPos[0].antPosZ = 0;
    tsArgs.txAntPos[1].antPosX = 0;
    tsArgs.txAntPos[1].antPosZ = 0;
    tsArgs.txAntPos[2].antPosX = 0;
    tsArgs.txAntPos[2].antPosZ = 0;

    PRINT_FUNC("Calling rlSetTestSourceConfig with Simulated Object at X[%d]cm, Y[%d]cm, Z[%d]cm \n\n",
            tsArgs.testObj[0].posX, tsArgs.testObj[0].posY, tsArgs.testObj[0].posZ);

    retVal = rlSetTestSourceConfig(deviceMap, &tsArgs);

    tsEnableArgs.tsEnable = 1U;
    tsEnableArgs.tsMode = 1U;
    retVal = rlTestSourceEnable(deviceMap, &tsEnableArgs);

    return retVal;
}
#endif

/** @fn int MMWL_dataPathConfig(unsigned char deviceMap)
*
*   @brief Data path configuration API. Configures CQ data size on the
*           lanes and number of samples of CQ[0-2] to br transferred.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Data path configuration API. Configures CQ data size on the
*   lanes and number of samples of CQ[0-2] to br transferred.
*/
int MMWL_dataPathConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDevDataPathCfg_t dataPathCfgArgs = gMmwSensCfg.datapathCfg.devDataPathCfg;

    PRINT_FUNC("Calling rlDeviceSetDataPathConfig with HSI Interface[%d] Selected \n\n",
            dataPathCfgArgs.intfSel);

    /* same API is used to configure CQ data size on the
     * lanes and number of samples of CQ[0-2] to br transferred.
     */
    retVal = rlDeviceSetDataPathConfig(deviceMap, &dataPathCfgArgs);
    return retVal;
}

/** @fn int MMWL_lvdsLaneConfig(unsigned char deviceMap)
*
*   @brief Lane Config API
*
*   @return Success - 0, Failure - Error Code
*
*   Lane Config API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_lvdsLaneConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDevLvdsLaneCfg_t lvdsLaneCfgArgs = gMmwSensCfg.datapathCfg.devLvdsLaneCfg;

    retVal = rlDeviceSetLvdsLaneConfig(deviceMap, &lvdsLaneCfgArgs);
    return retVal;
}

/** @fn int MMWL_laneConfig(unsigned char deviceMap)
*
*   @brief Lane Enable API
*
*   @return Success - 0, Failure - Error Code
*
*   Lane Enable API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_laneConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDevLaneEnable_t laneEnCfgArgs = gMmwSensCfg.datapathCfg.devLaneEnCfg;

    retVal = rlDeviceSetLaneConfig(deviceMap, &laneEnCfgArgs);
    return retVal;
}

/** @fn int MMWL_hsiLaneConfig(unsigned char deviceMap)
*
*   @brief LVDS lane configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   LVDS lane configuration API.
*/
int MMWL_hsiLaneConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    /*lane configuration*/
    retVal = MMWL_laneConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("LaneConfig failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("LaneConfig success for deviceMap %u\n\n", deviceMap);
    }
    /*LVDS lane configuration*/
    retVal = MMWL_lvdsLaneConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("LvdsLaneConfig failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("LvdsLaneConfig success for deviceMap %u\n\n", deviceMap);
    }
    return retVal;
}

/** @fn int MMWL_setHsiClock(unsigned char deviceMap)
*
*   @brief High Speed Interface Clock Config API
*
*   @return Success - 0, Failure - Error Code
*
*   HSI Clock Config API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_setHsiClock(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDevHsiClk_t hsiClkgs = gMmwSensCfg.datapathCfg.devHsiClkCfg;

    PRINT_FUNC("Calling rlDeviceSetHsiClk with HSI Clock[%d] \n\n",
            hsiClkgs.hsiClk);

    retVal = rlDeviceSetHsiClk(deviceMap, &hsiClkgs);
    return retVal;
}

/** @fn int MMWL_hsiDataRateConfig(unsigned char deviceMap)
*
*   @brief LVDS/CSI2 Clock Config API
*
*   @return Success - 0, Failure - Error Code
*
*   LVDS/CSI2 Clock Config API
*/
/* SourceId :  */
/* DesignId :  */
/* Requirements :  */
int MMWL_hsiDataRateConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDevDataPathClkCfg_t dataPathClkCfgArgs = gMmwSensCfg.datapathCfg.devDataPathClkCfg;

    PRINT_FUNC("Calling rlDeviceSetDataPathClkConfig with HSI Data Rate[%d] Selected \n\n",
            dataPathClkCfgArgs.dataRate);

    retVal = rlDeviceSetDataPathClkConfig(deviceMap, &dataPathClkCfgArgs);
    return retVal;
}

/** @fn int MMWL_hsiClockConfig(unsigned char deviceMap)
*
*   @brief Clock configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Clock configuration API.
*/
int MMWL_hsiClockConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK, readAllParams = 0;

    /*LVDS clock configuration*/
    retVal = MMWL_hsiDataRateConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("LvdsClkConfig failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("MMWL_hsiDataRateConfig success for deviceMap %u\n\n", deviceMap);
    }

    /*set high speed clock configuration*/
    retVal = MMWL_setHsiClock(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("MMWL_setHsiClock failed for deviceMap %u with error code %d\n\n",
                deviceMap, retVal);
        return -1;
    }
    else
    {
        PRINT_FUNC("MMWL_setHsiClock success for deviceMap %u\n\n", deviceMap);
    }

    return retVal;
}

/** @fn int MMWL_gpadcMeasConfig(unsigned char deviceMap)
*
*   @brief API to set GPADC configuration.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code.
*
*   API to set GPADC Configuration. And device will    send GPADC
*    measurement data in form of Asynchronous event over SPI to
*    Host. User needs to feed input signal on the device pins where
*    they want to read the measurement data inside the device.
*/
int MMWL_gpadcMeasConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    int timeOutCnt = 0;
    rlGpAdcCfg_t gpadcCfg = {0};

    /* enable all the sensors [0-6] to read gpADC measurement data */
    gpadcCfg.enable = 0x3F;
    /* set the number of samples device needs to collect to do the measurement */
    gpadcCfg.numOfSamples[0].sampleCnt = 32;
    gpadcCfg.numOfSamples[1].sampleCnt = 32;
    gpadcCfg.numOfSamples[2].sampleCnt = 32;
    gpadcCfg.numOfSamples[3].sampleCnt = 32;
    gpadcCfg.numOfSamples[4].sampleCnt = 32;
    gpadcCfg.numOfSamples[5].sampleCnt = 32;
    gpadcCfg.numOfSamples[6].sampleCnt = 32;

    retVal = rlSetGpAdcConfig(deviceMap, &gpadcCfg);

    if(retVal == RL_RET_CODE_OK)
    {
        while (mmwl_bGpadcDataRcv == 0U)
        {
            osiSleep(1); /*Sleep 1 msec*/
            timeOutCnt++;
            if (timeOutCnt > MMWL_API_RF_INIT_TIMEOUT)
            {
                retVal = RL_RET_CODE_RESP_TIMEOUT;
                break;
            }
        }
    }

    return retVal;
}

/** @fn int MMWL_sensorStart(unsigned char deviceMap)
*
*   @brief API to Start sensor.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   API to Start sensor.
*/
int MMWL_sensorStart(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    int timeOutCnt = 0;
    mmwl_bSensorStarted = 0U;
	
	/* trigger the DCA1000 Capture before frame trigger */
	mmw_TriggerDcaCapture();

    retVal = rlSensorStart(deviceMap);
#ifndef ENABLE_TEST_SOURCE
    if (((rfDevCfg.aeControl & 0x1) == 0x0) && (retVal == 0))
    {
        while (mmwl_bSensorStarted == 0U)
        {
            osiSleep(1); /*Sleep 1 msec*/
            timeOutCnt++;
            if (timeOutCnt > MMWL_API_RF_INIT_TIMEOUT)
            {
                retVal = RL_RET_CODE_RESP_TIMEOUT;
                break;
            }
        }
    }
#endif
    return retVal;
}

/** @fn int MMWL_sensorStop(unsigned char deviceMap)
*
*   @brief API to Stop sensor.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   API to Stop Sensor.
*/
int MMWL_sensorStop(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK, timeOutCnt =0;
    retVal = rlSensorStop(deviceMap);
#ifndef ENABLE_TEST_SOURCE
    if (retVal == RL_RET_CODE_OK)
    {
        if ((rfDevCfg.aeControl & 0x2) == 0x0)
        {
            while (mmwl_bSensorStarted == 1U)
            {
                osiSleep(1); /*Sleep 1 msec*/
                timeOutCnt++;
                if (timeOutCnt > MMWL_API_RF_INIT_TIMEOUT)
                {
                    retVal = RL_RET_CODE_RESP_TIMEOUT;
                    break;
                }
            }
        }
    }
#endif
    return retVal;
}

/** @fn int MMWL_setContMode(unsigned char deviceMap)
*
*   @brief API to set continuous mode.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   API to set continuous mode.
*/
int MMWL_setContMode(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlContModeCfg_t contModeCfgArgs = { 0 };
	contModeCfgArgs = gMmwSensCfg.rfCfg.contModeCfg;

    PRINT_FUNC("Calling setContMode with\n digOutSampleRate[%d]\nstartFreqConst[%d]\ntxOutPowerBackoffCode[%d]\nRXGain[%d]\n\n", \
        contModeCfgArgs.digOutSampleRate, contModeCfgArgs.startFreqConst, contModeCfgArgs.txOutPowerBackoffCode, \
        contModeCfgArgs.rxGain);
    retVal = rlSetContModeConfig(deviceMap, &contModeCfgArgs);
    return retVal;
}

/** @fn int MMWL_dynChirpEnable(unsigned char deviceMap)
*
*   @brief API to enable Dynamic chirp feature.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   API to enable Dynamic chirp feature.
* @Note : Not Supported in this application.
*/
int MMWL_dynChirpEnable(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    rlDynChirpEnCfg_t dynChirpEnCfgArgs = { 0 };

    retVal = rlSetDynChirpEn(deviceMap, &dynChirpEnCfgArgs);
    return retVal;
}

/** @fn int MMWL_dynChirpConfig(unsigned char deviceMap)
*
*   @brief API to config chirp dynamically.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   API to config chirp dynamically.
*  @Note: Not supported in this application
*/
int  MMWL_setDynChirpConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    unsigned int cnt;
    rlDynChirpCfg_t * dataDynChirp[3U] = { &dynChirpCfgArgs[0], &dynChirpCfgArgs[1], &dynChirpCfgArgs[2]};

    dynChirpCfgArgs[0].programMode = 0;

    /* Configure NR1 for 48 chirps */
    dynChirpCfgArgs[0].chirpRowSelect = 0x10;
    dynChirpCfgArgs[0].chirpSegSel = 0;
    /* Copy this dynamic chirp config to other config and update chirp segment number */
    memcpy(&dynChirpCfgArgs[1], &dynChirpCfgArgs[0], sizeof(rlDynChirpCfg_t));
    memcpy(&dynChirpCfgArgs[2], &dynChirpCfgArgs[0], sizeof(rlDynChirpCfg_t));
    /* Configure NR2 for 48 chirps */
    dynChirpCfgArgs[1].chirpRowSelect = 0x20;
    dynChirpCfgArgs[1].chirpSegSel = 1;
    /* Configure NR3 for 48 chirps */
    dynChirpCfgArgs[2].chirpRowSelect = 0x30;
    dynChirpCfgArgs[2].chirpSegSel = 2;

    for (cnt = 0; cnt < 16; cnt++)
    {
        /* Reconfiguring frequency slope for 48 chirps */
        dynChirpCfgArgs[0].chirpRow[cnt].chirpNR1 |= (((3*cnt) & 0x3FU) << 8);
        dynChirpCfgArgs[0].chirpRow[cnt].chirpNR2 |= (((3*cnt + 1) & 0x3FU) << 8);
        dynChirpCfgArgs[0].chirpRow[cnt].chirpNR3 |= (((3*cnt + 2) & 0x3FU) << 8);
        /* Reconfiguring start frequency for 48 chirps */
        dynChirpCfgArgs[1].chirpRow[cnt].chirpNR1 |= 3*cnt;
        dynChirpCfgArgs[1].chirpRow[cnt].chirpNR2 |= 3*cnt + 1;
        dynChirpCfgArgs[1].chirpRow[cnt].chirpNR3 |= 3*cnt + 2;
        /* Reconfiguring ideal time for 48 chirps */
        dynChirpCfgArgs[2].chirpRow[cnt].chirpNR1 |= 3 * cnt;
        dynChirpCfgArgs[2].chirpRow[cnt].chirpNR2 |= 3 * cnt + 1;
        dynChirpCfgArgs[2].chirpRow[cnt].chirpNR3 |= 3 * cnt + 2;
    }

    PRINT_FUNC("Calling DynChirpCfg with chirpSegSel[%d]\nchirpNR1[%d]\n\n", \
        dynChirpCfgArgs[0].chirpSegSel, dynChirpCfgArgs[0].chirpRow[0].chirpNR1);
    retVal = rlSetDynChirpCfg(deviceMap, 2U, &dataDynChirp[0]);
    return retVal;
}

/** @fn int MMWL_rlRfAnaMonConfig(unsigned char deviceMap)
*
*   @brief consolidated configuration of all ana mon.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Sets the consolidated configuration of all analog monitoring..
*/
int MMWL_MonitoringConfig(unsigned char deviceMap)
{
	int status = 0, retVal = 0, switch_on;
	rlRfCalMonTimeUntConf_t calMonTimeUnitCfg = { 0 };
	rlRunTimeCalibConf_t	runTimeCalibCfg = { 0 };
	rlRfCalMonFreqLimitConf_t calMonFreqLimitCfg = { 0 };
	rlMonAnaEnables_t rlAnaMonCnfgArgs = { 0 };

	if (memcmp(&gMmwSensCfg.rfCfg.calMonTimeUntCfg, &calMonTimeUnitCfg, sizeof(rlRfCalMonTimeUntConf_t)) != 0)
	{
		retVal = rlRfSetCalMonTimeUnitConfig(deviceMap, &gMmwSensCfg.rfCfg.calMonTimeUntCfg);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("CalMonTimeUnit Configuration failed for deviceMap %u with error code %d \n\n", \
				deviceMap, retVal);
			return retVal;
		}
		else
		{
			PRINT_FUNC("Monitoring Configuration successful for deviceMap %u \n\n", deviceMap);
		}
	}

	if (memcmp(&gMmwSensCfg.rfCfg.runTimeCalCfg, &runTimeCalibCfg, sizeof(rlRunTimeCalibConf_t)) != 0)
	{
		retVal = rlRfRunTimeCalibConfig(deviceMap, &gMmwSensCfg.rfCfg.runTimeCalCfg);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("rlRfRunTimeCalibConfig failed for deviceMap %u with error code %d \n\n", \
				deviceMap, retVal);
			return retVal;
		}
		else
		{
			PRINT_FUNC("rlRfRunTimeCalibConfig successful for deviceMap %u \n\n", deviceMap);
		}
	}

	if (memcmp(&gMmwSensCfg.rfCfg.rfCalMonFreqLimitCfg, &calMonFreqLimitCfg, \
		sizeof(rlRfCalMonFreqLimitConf_t)) != 0)
	{
		retVal = rlRfSetCalMonFreqLimitConfig(deviceMap, &gMmwSensCfg.rfCfg.rfCalMonFreqLimitCfg);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("rlRfSetCalMonFreqLimitConfig failed for deviceMap %u with error code %d \n\n", \
				deviceMap, retVal);
			return retVal;
		}
		else
		{
			PRINT_FUNC("rlRfSetCalMonFreqLimitConfig successful for deviceMap %u \n\n", deviceMap);
		}
	}

	for (switch_on = 0; switch_on <= RX_MIXER_INPUT_POWER_MONITOR; switch_on++)
	{
		if (!(gMmwSensCfg.monitorCfg.anaMonEnCfg.enMask & (1 << switch_on)))
		{
			continue;
		}

		switch (switch_on)
		{
			case TEMPERATURE_MONITOR_EN:
				retVal = rlRfTempMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.tempMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfTempMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfTempMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case RX_GAIN_PHASE_MONITOR_EN:
				retVal = rlRfRxGainPhMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.rxGainPhMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxGainPhMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxGainPhMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case RX_NOISE_MONITOR_EN:
				retVal = rlRfRxNoiseMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.rxNoiseMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxNoiseMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxNoiseMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case RX_IFSTAGE_MONITOR_EN:
				retVal = rlRfRxIfStageMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.rxIfStageMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxIfStageMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxIfStageMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case TX0_POWER_MONITOR_EN:
			case TX1_POWER_MONITOR_EN:
			case TX2_POWER_MONITOR_EN:
				retVal = rlRfTxPowrMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.allTxPowerMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfTxPowrMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfTxPowrMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case TX0_BALLBREAK_MONITOR_EN:
			case TX1_BALLBREAK_MONITOR_EN:
			case TX2_BALLBREAK_MONITOR_EN:
				retVal = rlRfTxBallbreakMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.allTxBallBreakMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfTxBallbreakMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfTxBallbreakMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case TX_GAIN_PHASE_MONITOR_EN:
				retVal = rlRfTxGainPhaseMismatchMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.txGainPhMisMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfTxGainPhaseMismatchMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfTxGainPhaseMismatchMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case TX0_BPM_MONITOR_EN:
			case TX1_BPM_MONITOR_EN:
			case TX2_BPM_MONITOR_EN:
				/* Not Supported */
				break;
			case SYNTH_FREQ_MONITOR_EN:
				retVal = rlRfSynthFreqMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.synthFreqMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfSynthFreqMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfSynthFreqMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case EXTERNAL_ANALOG_SIGNALS_MONITOR_EN:
				retVal = rlRfExtAnaSignalsMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.extAnaSigMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfExtAnaSignalsMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfExtAnaSignalsMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case INTERNAL_TX0_SIGNALS_MONITOR_EN:
			case INTERNAL_TX1_SIGNALS_MONITOR_EN:
			case INTERNAL_TX2_SIGNALS_MONITOR_EN:
				retVal = rlRfTxIntAnaSignalsMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.allTxIntAnaSigMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfTxIntAnaSignalsMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfTxIntAnaSignalsMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case INTERNAL_RX_SIGNALS_MONITOR_EN:
				retVal = rlRfRxIntAnaSignalsMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxIntAnaSignalsMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxIntAnaSignalsMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case INTERNAL_PMCLKLO_SIGNALS_MONITOR_EN:
				retVal = rlRfPmClkLoIntAnaSignalsMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.pmClkAnaSigMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfPmClkLoIntAnaSignalsMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfPmClkLoIntAnaSignalsMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case INTERNAL_GPADC_SIGNALS_MONITOR_EN:
				retVal = rlRfGpadcIntAnaSignalsMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.gpadcAnaSigMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfGpadcIntAnaSignalsMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfGpadcIntAnaSignalsMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case PLL_CONTROL_VOLTAGE_MONITOR_EN:
				retVal = rlRfPllContrlVoltMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.pllCtrlVoltMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfPllContrlVoltMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfPllContrlVoltMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case DCC_CLOCK_FREQ_MONITOR_EN:
				retVal = rlRfDualClkCompMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.dualClkComMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfDualClkCompMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfDualClkCompMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case RX_IF_SATURATION_MONITOR_EN:
				retVal = rlRfRxIfSatMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.rxSatMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxIfSatMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxIfSatMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case RX_SIG_IMG_BAND_MONITORING_EN:
				retVal = rlRfRxSigImgMonConfig(deviceMap, &gMmwSensCfg.monitorCfg.sigImgMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxSigImgMonConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxSigImgMonConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			case RX_MIXER_INPUT_POWER_MONITOR:
				retVal = rlRfRxMixerInPwrConfig(deviceMap, &gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg);
				if (retVal != RL_RET_CODE_OK)
				{
					PRINT_FUNC("rlRfRxMixerInPwrConfig failed for deviceMap %u with error code %d \n\n", \
						deviceMap, retVal);
					return retVal;
				}
				else
				{
					PRINT_FUNC("rlRfRxMixerInPwrConfig successful for deviceMap %u \n\n", deviceMap);
				}
				break;
			default:
				break;
		}
	}

	if (memcmp(&gMmwSensCfg.monitorCfg.anaMonEnCfg, &rlAnaMonCnfgArgs, \
		sizeof(rlMonAnaEnables_t)) != 0)
	{
		/* Analog monitoring configuration */
		retVal = rlRfAnaMonConfig(RL_DEVICE_MAP_CASCADED_1, &gMmwSensCfg.monitorCfg.anaMonEnCfg);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("rlRfAnaMonConfig failed for deviceMap %u with error code %d \n\n", \
				deviceMap, retVal);
			return retVal;
		}
		else
		{
			PRINT_FUNC("rlRfAnaMonConfig successful for deviceMap %u \n\n", deviceMap);
		}
	}

	return retVal;
}

/** @fn int MMWL_powerOff(unsigned char deviceMap)
*
*   @brief API to poweroff device.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   API to poweroff device.
*/
int MMWL_powerOff(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    retVal = rlDevicePowerOff();
    mmwl_bInitComp = 0;
    mmwl_bStartComp = 0U;
    mmwl_bRfInitComp = 0U;
    mmwl_devHdl = NULL;

    return retVal;
}

/** @fn int MMWL_lowPowerConfig(deviceMap)
*
*   @brief LowPower configuration API.
*
*   @param[in] deviceMap - Devic Index
*
*   @return int Success - 0, Failure - Error Code
*
*   LowPower configuration API.
*/
int MMWL_lowPowerConfig(unsigned char deviceMap)
{
    int retVal = RL_RET_CODE_OK;
    /* TBD - Read GUI Values */
    rlLowPowerModeCfg_t rfLpModeCfgArgs = gMmwSensCfg.rfCfg.lowPowerCfg;

    retVal = rlSetLowPowerModeConfig(deviceMap, &rfLpModeCfgArgs);
    return retVal;
}

/** @fn int MMWL_SOPControl(unsigned char deviceMap, int SOPmode)
*
*   @brief SOP mode configuration API.
*
*   @param[in] deviceMap - Device Index
*
*   @return int Success - 0, Failure - Error Code
*
*   SOP mode configuration API.
*/
int MMWL_SOPControl(unsigned char deviceMap, int SOPmode)
{
	int retVal = RL_RET_CODE_OK;
	rlsDevHandle_t rlImpl_devHdl = NULL;

	rlImpl_devHdl = rlsGetDeviceCtx(0);
	if (rlImpl_devHdl != NULL)
	{
		retVal = rlsOpenGenericGpioIf(rlImpl_devHdl);
		retVal = rlsSOPControl(rlImpl_devHdl, SOPmode);
		retVal = rlsOpenBoardControlIf(rlImpl_devHdl);
	}
	else
	{
		retVal = RL_RET_CODE_INVALID_STATE_ERROR;
	}
	return retVal;
}

/** @fn int MMWL_ResetDevice(unsigned char deviceMap)
*
*   @brief Device Reset configuration API.
*
*   @param[in] deviceMap - Device Index
*
*   @return int Success - 0, Failure - Error Code
*
*   Device Reset configuration API.
*/
int MMWL_ResetDevice(unsigned char deviceMap)
{
	int retVal = RL_RET_CODE_OK;
	/* Reset the devices */
	rlsDevHandle_t rlImpl_devHdl = NULL;

	rlImpl_devHdl = rlsGetDeviceCtx(0);
	if (rlImpl_devHdl != NULL)
	{
		retVal = rlsFullReset(rlImpl_devHdl, 0);
		retVal = rlsFullReset(rlImpl_devHdl, 1);
		retVal = rlsCloseBoardControlIf(rlImpl_devHdl);
		retVal = rlsCloseGenericGpioIf(rlImpl_devHdl);
	}
	else
	{
		retVal = RL_RET_CODE_INVALID_STATE_ERROR;
	}
	return retVal;
}

/** @fn int MMWL_App()
*
*   @brief mmWaveLink Example Application.
*
*   @return int Success - 0, Failure - Error Code
*
*   mmWaveLink Example Application.
*/
int MMWL_App(int deviceType)
{
    int retVal = RL_RET_CODE_OK;
    unsigned char deviceMap = RL_DEVICE_MAP_CASCADED_1;
	int SOPmode = 0;

	gDeviceType = rlDevGlobalCfgArgs.mmwaveDevVariant;
	gCalMonTimeUnit = gMmwSensCfg.rfCfg.calMonTimeUntCfg.calibMonTimeUnit;
	rlDevGlobalCfgArgs.LinkAdvChirpTest = FALSE;

	if (rlDevGlobalCfgArgs.TransferMode == 0)
	{
		PRINT_FUNC("====================== SPI Mode of Operation ======================\n\n");
		SOPmode = 4; /* Set SOP 4 mode for SPI */
	}
	else if (rlDevGlobalCfgArgs.TransferMode == 1)
	{
		PRINT_FUNC("====================== I2C Mode of Operation ======================\n\n");
		SOPmode = 7; /* Set SOP 7 mode for I2C */
	}
	else
	{
		PRINT_FUNC("Invalid Transport Mode detected with transportMode %d\n\n", rlDevGlobalCfgArgs.TransferMode);
		return -1;
	}

	retVal = MMWL_SOPControl(deviceMap, SOPmode);
	if (retVal != RL_RET_CODE_OK)
	{
		PRINT_FUNC("Device map %u : SOP %d mode failed with error %d\n\n", deviceMap, SOPmode, retVal);
		return retVal;
	}
	else
	{
		PRINT_FUNC("Device map %u : SOP %d mode successful\n\n", deviceMap, SOPmode);
	}

	retVal = MMWL_ResetDevice(deviceMap);
	if (retVal != RL_RET_CODE_OK)
	{
		PRINT_FUNC("Device map %u : Device reset failed with error %d \n\n", deviceMap,
			retVal);
		return retVal;
	}
	else
	{
		PRINT_FUNC("Device map %u : Device reset successful\n\n", deviceMap);
	}

    /*  \subsection     api_sequence1     Seq 1 - Call Power ON API
    The mmWaveLink driver initializes the internal components, creates Mutex/Semaphore,
    initializes buffers, register interrupts, bring mmWave front end out of reset.
    */
    retVal = MMWL_powerOnMaster(deviceMap, deviceType);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("mmWave Device Power on failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("mmWave Device Power on success for deviceMap %u \n\n",
                deviceMap);
    }

    /*  \subsection     api_sequence2     Seq 2 - Download Firmware/patch (Optional)
    The mmWave device firmware is ROMed and also can be stored in External Flash. This
    step is necessary if firmware needs to be patched and patch is not stored in serial
    Flash
    */
	if (!rlDevGlobalCfgArgs.DisableFwDownload)
	{
		PRINT_FUNC("========================== Firmware Download ==========================\n\n");
		retVal = MMWL_firmwareDownload(deviceMap);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("Firmware update failed for deviceMap %u with error %d \n\n",
				deviceMap, retVal);
			return retVal;
		}
		else
		{
			PRINT_FUNC("Firmware update successful for deviceMap %u \n\n",
				deviceMap);
		}
		PRINT_FUNC("=====================================================================\n\n");
	}
	if (deviceType == AWR2243)
	{
		/* Swap reset and power on the device */
		retVal = MMWL_SwapResetAndPowerOn(deviceMap);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("Could not restart the device\n\n");
			return retVal;
		}
	}

    /* Change CRC Type of Async Event generated by MSS to what is being requested by user in mmwaveconfig.txt */
    retVal = MMWL_setDeviceCrcType(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("CRC Type set for MasterSS failed for deviceMap %u with error code %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("CRC Type set for MasterSS success for deviceMap %u \n\n", deviceMap);
    }

    /*  \subsection     api_sequence3     Seq 3 - Enable the mmWave Front end (Radar/RF subsystem)
    The mmWave Front end once enabled runs boot time routines and upon completion sends asynchronous event
    to notify the result
    */
    retVal = MMWL_rfEnable(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Radar/RF subsystem Power up failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("Radar/RF subsystem Power up successful for deviceMap %u \n\n", deviceMap);
    }

    /*  \subsection     api_sequence4     Seq 4 - Basic/Static Configuration
    The mmWave Front end needs to be configured for mmWave Radar operations. basic
    configuration includes Rx/Tx channel configuration, ADC configuration etc
    */
    PRINT_FUNC("======================Basic/Static Configuration======================\n\n");
    retVal = MMWL_basicConfiguration(deviceMap, 0);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Basic/Static configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("Basic/Static configuration success for deviceMap %u \n\n",
                deviceMap);
    }

    /*  \subsection     api_sequence5     Seq 5 - Initializes the mmWave Front end
    The mmWave Front end once configured needs to be initialized. During initialization
    mmWave Front end performs calibration and once calibration is complete, it
    notifies the application using asynchronous event
    */
    retVal = MMWL_rfInit(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("RF Initialization/Calibration failed for deviceMap %u with error code %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("RF Initialization/Calibration successful for deviceMap %u \n\n", deviceMap);
    }

	/*  \subsection     api_sequence6     Seq 6 - Configures the programmable filter */
	retVal = MMWL_progFiltConfig(deviceMap);

    /*  \subsection     api_sequence8     Seq 8 - FMCW profile configuration
    TI mmWave devices supports Frequency Modulated Continuous Wave(FMCW) Radar. User
    Need to define characteristics of FMCW signal using profile configuration. A profile
    contains information about FMCW signal such as Start Frequency (76 - 81 GHz), Ramp
    Slope (e.g 30MHz/uS). Idle Time etc. It also configures ADC samples, Sampling rate,
    Receiver gain, Filter configuration parameters

    \ Note - User can define upto 4 different profiles
    */
    PRINT_FUNC("======================FMCW Configuration======================\n\n");
    retVal = MMWL_profileConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Profile Configuration failed for deviceMap %u with error code %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("Profile Configuration success for deviceMap %u \n\n", deviceMap);
    }

	if (rlDevGlobalCfgArgs.LinkAdvChirpTest == FALSE)
	{
		/*  \subsection     api_sequence9     Seq 9 - FMCW chirp configuration
		A chirp is always associated with FMCW profile from which it inherits coarse information
		about FMCW signal. Using chirp configuration user can further define fine
		variations to coarse parameters such as Start Frequency variation(0 - ~400 MHz), Ramp
		Slope variation (0 - ~3 MHz/uS), Idle Time variation etc. It also configures which transmit channels to be used
		for transmitting FMCW signal.

		\ Note - User can define upto 512 unique chirps
		*/
		retVal = MMWL_chirpConfig(deviceMap);
		if (retVal != RL_RET_CODE_OK)
		{
			PRINT_FUNC("Chirp Configuration failed for deviceMap %u with error %d \n\n",
				deviceMap, retVal);
			return retVal;
		}
		else
		{
			PRINT_FUNC("Chirp Configuration success for deviceMap %u \n\n", deviceMap);
		}
	}
	else
	{

	}

    /*  \subsection     api_sequence10     Seq 10 - Data Path (CSI2/LVDS) Configuration
    TI mmWave device supports CSI2 or LVDS interface for sending RAW ADC data. mmWave device
    can also send Chirp Profile and Chirp Quality data over LVDS/CSI2. User need to select
    the high speed interface and also what data it expects to receive.

    \ Note - This API is only applicable for AWR2243 when mmWaveLink driver is running on External Host
    */
    PRINT_FUNC("==================Data Path(LVDS/CSI2) Configuration==================\n\n");
    retVal = MMWL_dataPathConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Data Path Configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("Data Path Configuration successful for deviceMap %u \n\n", deviceMap);
    }

    /*  \subsection     api_sequence11     Seq 11 - CSI2/LVDS CLock and Data Rate Configuration
    User need to configure what data rate is required to send the data on high speed interface. For
    e.g 150mbps, 300mbps etc.
    \ Note - This API is only applicable for AWR2243 when mmWaveLink driver is running on External Host
    */
    retVal = MMWL_hsiClockConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("CSI2/LVDS Clock Configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("CSI2/LVDS Clock Configuration success for deviceMap %u \n\n", deviceMap);
    }

    /*  \subsection     api_sequence12     Seq 12 - CSI2/LVDS Lane Configuration
    User need to configure how many LVDS/CSI2 lane needs to be enabled
    \ Note - This API is only applicable for AWR2243 when mmWaveLink driver is running on External Host
    */
    retVal = MMWL_hsiLaneConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("CSI2/LVDS Lane Config failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("CSI2/LVDS Lane Configuration success for deviceMap %u \n\n",
                deviceMap);
    }
    PRINT_FUNC("======================================================================\n\n");

#ifdef ENABLE_TEST_SOURCE
    retVal = MMWL_testSourceConfig(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Test Source Configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("Test source Configuration success for deviceMap %u \n\n", deviceMap);
    }
#endif

	/* Check for If Advance Frame Test is enabled in JSON,
	If Legacy Frame Config comes first in JSON file it assumes it to legacy frame based. */
    if(gMmwSensCfg.rfCfg.dfeOutputMode == MMW_LEFACY_FRAME_MODE)
		//rlDevGlobalCfgArgs.LinkAdvanceFrameTest == FALSE)
    {
        /*  \subsection     api_sequence13     Seq 13 - FMCW frame configuration
        A frame defines sequence of chirps and how this sequence needs to be repeated over time.
        */
        retVal = MMWL_frameConfig(deviceMap);
        if (retVal != RL_RET_CODE_OK)
        {
            PRINT_FUNC("Frame Configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
            return retVal;
        }
        else
        {
            PRINT_FUNC("Frame Configuration success for deviceMap %u \n\n", deviceMap);
        }
    }
    else if (gMmwSensCfg.rfCfg.dfeOutputMode == MMW_ADV_FRAME_MODE)
    {
        /*  \subsection     api_sequence14     Seq 14 - FMCW Advance frame configuration
        A frame defines sequence of chirps and how this sequence needs to be repeated over time.
        */
        retVal = MMWL_advFrameConfig(deviceMap);

        if (retVal != RL_RET_CODE_OK)
        {
            PRINT_FUNC("Adv Frame Configuration failed for deviceMap %u with error %d \n\n",
                deviceMap, retVal);
            return retVal;
        }
        else
        {
            PRINT_FUNC("Adv Frame Configuration success for deviceMap %u \n\n", deviceMap);
        }
    }
	/* continuous Mode */
	else
	{
        retVal = MMWL_setContMode(deviceMap);
        if (retVal != RL_RET_CODE_OK)
        {
            PRINT_FUNC("Continuous mode Config failed for deviceMap %u with error code %d \n\n",
                deviceMap, retVal);
            return retVal;
        }
        else
        {
            PRINT_FUNC("Continuous mode Config successful for deviceMap %u \n\n", deviceMap);
        }
    }

	/* enable Monitoring APIs */
	retVal = MMWL_MonitoringConfig(deviceMap);
	if (retVal != RL_RET_CODE_OK)
	{
		PRINT_FUNC("Monitoring Configuration failed for deviceMap %u with error code %d \n\n",\
			deviceMap, retVal);
		return retVal;
	}
	else
	{
		PRINT_FUNC("Monitoring Configuration successful for deviceMap %u \n\n", deviceMap);
	}

    /*  \subsection     api_sequence15     Seq 15 - Start mmWave Radar Sensor
    This will trigger the mmWave Front to start transmitting FMCW signal. Raw ADC samples
    would be received from Digital front end. For AWR2243, if high speed interface is
    configured, RAW ADC data would be transmitted over CSI2/LVDS. On xWR1443/xWR1642, it can
    be processed using HW accelerator or DSP
    */
    retVal = MMWL_sensorStart(deviceMap);
    if (retVal != RL_RET_CODE_OK)
    {
        PRINT_FUNC("Sensor Start failed for deviceMap %u with error code %d \n\n",
                deviceMap, retVal);
        return retVal;
    }
    else
    {
        PRINT_FUNC("Sensor Start successful for deviceMap %u \n\n", deviceMap);
    }

    return 0;
}

