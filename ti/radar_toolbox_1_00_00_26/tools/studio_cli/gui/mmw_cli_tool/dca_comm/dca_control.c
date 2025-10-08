/****************************************************************************************
* FileName     : dca_control.c
*
* Description  : This file implements the DCA1000 features to configure and control it.
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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rf_api.h"
#include "mmw_config.h"

#define DCA1000_CAPTURE_ONGOING            1
#define DCA1000_CAPTURE_DONE               2
#define DCA1000_CAPTURE_STOP               3

/** SUCCESS status (generic) */
#define SUCCESS_STATUS                      0

/** Failure status (generic) */
#define FAILURE_STATUS                      1

/** Fpga config - Timer value   */
#define FPGA_CONFIG_DEFAULT_TIMER           30

/** CLI command timeout duration in millisec */
#define CLI_CMD_TIMEOUT_DURATION            7000

/** Shared memory polling frequency in millisec for record status */
#define MILLI_SEC_TO_READ_SHM               500

/** Delay before printing stop record status */
#define MIN_MILLI_SEC_SLEEP_TO_DISP         100

/** CLI log file name   */
#define CLI_LOG_NAME                    "CLI_LogFile.txt"

/** FPGA config mode stucture object            */
strFpgaConfigMode gsFpgaConfigMode;

/** Ethernet config mode stucture object        */
strEthConfigMode gsEthConfigMode;

/** Record config mode stucture object          */
strRecConfigMode gsRecConfigMode;

/** Start record config mode stucture object    */
strStartRecConfigMode gsStartRecConfigMode;

/** Quiet mode enable/disable                   */
BOOL gbCliQuietMode = FALSE;

volatile char dcaCaptureMode = 0;

/** @fn void WRITE_TO_CONSOLE(const SINT8 *msg)
 * @brief This function is to write the CLI messages in the console  <!--
 * --> if quiet mode is not enabled
 * @param [in] msg [const SINT8 *] - Message to display
 * @return SINT32 value
 */
void WRITE_TO_CONSOLE(const SINT8 *msg)
{
    if(!gbCliQuietMode)
    {
		PRINT_FUNC("\n%s\n",msg);
    }
}

/** @fn void WRITE_TO_LOG_FILE(const SINT8 *s8Msg)
 * @brief This function is to write the CLI messages in the logfile
 * @param [in] s8Msg [const SINT8 *] - Message to display
 */
void WRITE_TO_LOG_FILE
(
    const SINT8 *s8Msg
)
{
    time_t loggingTime = time(NULL);
    FILE * pDebugFile;
    pDebugFile = fopen(CLI_LOG_NAME, "a+");

    if(NULL != pDebugFile)
    {
        fprintf(pDebugFile, "\n%s%s\n", ctime(&loggingTime), s8Msg);
    }

    fclose(pDebugFile);
}

/** @fn void EthernetEventHandlercallback(UINT16 u16CmdCode, UINT16 u16Status)
 * @brief RF data card capture event handler
 * @param [in] cmd code
 * @param [in] status
 */
void EthernetEventHandlercallback(UINT16 u16CmdCode, UINT16 u16Status)
{
    PRINT_FUNC("Call back called :  Async event recieved %d, %d \n", u16CmdCode, u16Status);

    //Syste Error cmd code
    if (u16CmdCode == 0x0A)
    {
        switch (u16Status)
        {
            case 0x0:
                PRINT_FUNC("STS_NO_LVDS_DATA Async event recieved [%d] \n",u16Status);
                break;
            case 0x1:
                PRINT_FUNC("STS_NO_HEADER Async event recieved [%d] \n",u16Status);
                break;
            case 0x2:
                PRINT_FUNC("STS_EEPROM_FAILURE Async event recieved [%d] \n",u16Status);
                break;
            case 0x3:
                PRINT_FUNC("STS_SD_CARD_DETECTED Async event recieved [%d] \n",u16Status);
                break;
            case 0x4:
                PRINT_FUNC("STS_SD_CARD_REMOVED Async event recieved [%d] \n",u16Status);
                break;
            case 0x5:
                PRINT_FUNC("STS_SD_CARD_FULL Async event recieved [%d] \n",u16Status);
                break;
            case 0x6:
                PRINT_FUNC("STS_MODE_CONFIG_FAILURE Async event recieved [%d] \n",u16Status);
                break;
            case 0x7:
                PRINT_FUNC("STS_DDR_FULL Async event recieved [%d] \n",u16Status);
                break;
            case 0x8:
				dcaCaptureMode = DCA1000_CAPTURE_DONE;
				/* for fixed set of frame count or on sensorStop, dca will generate this 
				 * event, so stop the dca capture which will update the adc*.bin file */
				mmw_CaptureDcaDone(); 
				PRINT_FUNC("STS_RECORD_COMPLETED Async event recieved [%d] \n",u16Status);
                break;
            case 0x9:
                PRINT_FUNC("STS_LVDS_BUFFER_FULL Async event recieved [%d] \n",u16Status);
                break;
            case 0xA:
				PRINT_FUNC("STS_PLAYBACK_COMPLETED Async event recieved [%d] \n",u16Status);
                break;
            case 0xB:
                PRINT_FUNC("STS_PLAYBACK_OUT_OF_SEQUENCE Async event recieved [%d] \n",u16Status);
                break;
            default:
                PRINT_FUNC("Syste Error cmd code :  Async event recieved [%d] \n",u16Status);
                break;
        }
    }
    else
    {
        switch (u16CmdCode)
        {
            //Event register status
            case 0x01:
                PRINT_FUNC("RESET_FPGA_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x02:
                PRINT_FUNC("RESET_AR_DEV_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x03:
                PRINT_FUNC("CONFIG_FPGA_GEN_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x04:
                PRINT_FUNC("CONFIG_EEPROM_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x05:
				dcaCaptureMode = DCA1000_CAPTURE_ONGOING;
                PRINT_FUNC("RECORD_START_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x06:
				dcaCaptureMode = DCA1000_CAPTURE_STOP;
                PRINT_FUNC("RECORD_STOP_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x07:
                PRINT_FUNC("PLAYBACK_START_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x08:
                PRINT_FUNC("PLAYBACK_STOP_CMD_CODE event recieved [%d] \n",u16CmdCode);
                break;
            case 0x09:
                PRINT_FUNC("SYSTEM_CONNECT_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x0A:
                PRINT_FUNC("SYSTEM_ERROR_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x0B:
                PRINT_FUNC("CONFIG_PACKET_DATA_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x0C:
                PRINT_FUNC("CONFIG_DATA_MODE_AR_DEV_CMD_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0x0D:
                PRINT_FUNC("INIT_FPGA_PLAYBACK_CMD_CODE event recieved [%d] \n",u16CmdCode);
                break;
            case 0x0E:
                break; 
            case 0xC1:
                PRINT_FUNC("INVALID_RESP_PKT_ERROR_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC2:
                PRINT_FUNC("RECORD_FILE_CREATION_ERROR_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC3:
                PRINT_FUNC("RECORD_PKT_OUT_OF_SEQ_ERROR_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC4:
                PRINT_FUNC("RECORD_IS_IN_PROGRESS_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC5:
                PRINT_FUNC("GUI_PLAYBACK_COMPLETED_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC6:
                PRINT_FUNC("PLAYBACK_FILE_OPEN_ERROR_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC7:
                PRINT_FUNC("PLAYBACK_UDP_WRITE_ERR Async event recieved [%d] \n",u16CmdCode);
                break;
            case 0xC8:
                PRINT_FUNC("PLAYBACK_IS_IN_PROGRESS_CODE Async event recieved [%d] \n",u16CmdCode);
                break;
            default:
                PRINT_FUNC("RF data capture related Async event recieved [%d] \n",u16CmdCode);
                break;
        }
                
    }
}

/** @fn void cli_DisconnectRFDCCard(UINT16 u16CmdCode)
 * @brief This function is to disconnect from DCA1000EVM sysytem
 * @param [in] u16CmdCode [UINT16] - Command code
 */
void DisconnectRFDCCard(UINT16 u16CmdCode)
{
    if(u16CmdCode == CMD_CODE_CLI_ASYNC_RECORD_STOP)
    {
        DisconnectRFDCCard_AsyncCommandMode();
    }
    else
    {
        DisconnectRFDCCard_ConfigMode();
    }
}

/** @fn void DecodeCommandStatus(SINT32 s32Status, const SINT8 * strCommand)
 * @brief This function is to decode command response status as success or failure
 * @param [in] s32Status [SINT32] - Command status
 * @param [in] strCommand [const SINT8 *] - Command name
 */
void DecodeCommandStatus(SINT32 s32Status, const SINT8 * strCommand)
{
    SINT8 msgData[MAX_NAME_LEN];

    if(s32Status == SUCCESS_STATUS)
    {
        sprintf(msgData, "%s command : Success",strCommand);
    }
    else if(s32Status == FAILURE_STATUS)
    {
        sprintf(msgData, "%s command : Failed",strCommand);
    }
    else if(s32Status == STS_RFDCCARD_INVALID_INPUT_PARAMS)
    {
        sprintf(msgData, "%s : \nVerify the input parameters - %d",
                strCommand, s32Status);
    }
    else if(s32Status == STS_RFDCCARD_OS_ERR)
    {
        sprintf(msgData, "%s : \nOS error - %d",
                strCommand, s32Status);
    }
    else if(s32Status == STS_RFDCCARD_UDP_WRITE_ERR)
    {
        sprintf(msgData, "%s : \nSending command failed - %d",
                strCommand, s32Status);
    }
    else if(s32Status == STS_RFDCCARD_TIMEOUT_ERR)
    {
        sprintf(msgData, "%s : \nTimeout Error! System disconnected",
                strCommand);
    }
    else if(s32Status == STS_INVALID_RESP_PKT_ERR)
    {
        sprintf(msgData, "%s : \nInvalid response packet error code",
                strCommand);
    }
    else
    {
        sprintf(msgData, "%s Cmd", strCommand);
    }

    WRITE_TO_CONSOLE(msgData);
    sprintf(msgData, "Return status : %d", s32Status);
}

/** @fn void LoadDcaGlobalConfigs(rlDcaConfig_t *dcaUserInputCfg)
 * @brief This function updates the global structs based on user config inputs.
 * @param [in] rlDcaConfig_t *dcaUserInputCfg user config input
 */
void LoadDcaGlobalConfigs(rlDcaConfig_t *dcaUserInputCfg)
{
	SINT8 *token;
	SINT32 k = 0;

	gsStartRecConfigMode.bMsbToggleEnable = FALSE;
	gsStartRecConfigMode.bReorderEnable   = FALSE;
	gsStartRecConfigMode.eConfigLogMode	  = RAW_MODE;
	gsStartRecConfigMode.eLvdsMode		  = (dcaUserInputCfg->lvdsLaneMode == 4) ? FOUR_LANE: TWO_LANE;
	gsStartRecConfigMode.eRecordStopMode  = (RecordStopMode)dcaUserInputCfg->captureStopMode;

	if(gsStartRecConfigMode.eRecordStopMode == BYTES)
	{
		gsStartRecConfigMode.u32BytesToCapture = dcaUserInputCfg->bytesToCapture;
	}
	else if(gsStartRecConfigMode.eRecordStopMode == DURATION)
	{
		gsStartRecConfigMode.u32DurationToCapture = dcaUserInputCfg->durationToCapture_ms;
	}
	else if(gsStartRecConfigMode.eRecordStopMode == FRAMES)
	{
		gsStartRecConfigMode.u32FramesToCapture = dcaUserInputCfg->framesToCapture;
	}
	else
	{
		/* nothing */
	}

	/* create directory if not exist */
	_mkdir(dcaUserInputCfg->adcDataPath);
	strcpy(gsStartRecConfigMode.s8FileBasePath, dcaUserInputCfg->adcDataPath);
	strcpy(gsStartRecConfigMode.s8FilePrefix, dcaUserInputCfg->filePrefix);
	gsStartRecConfigMode.u16MaxRecFileSize = dcaUserInputCfg->maxRecFileSize_MB;
	gsStartRecConfigMode.bSequenceNumberEnable = dcaUserInputCfg->sequenceNumberEnable;

	gsRecConfigMode.u16RecDelay = dcaUserInputCfg->packetDelay_us;

	/* Ethernet configuration */
	token = strtok(dcaUserInputCfg->DCA1000IPAddress, ".");
	k = 0;
	while (token != NULL) {
		gsEthConfigMode.au8Dca1000IpAddr[k++] = atoi(token);
		token = strtok(NULL, ".");
	}

	token = strtok(dcaUserInputCfg->DCA1000MacAddress, ".");
	k = 0;
	while (token != NULL) {
		gsEthConfigMode.au8MacId[k++] = atoi(token);
		token = strtok(NULL, ".");
	}

	token = strtok(dcaUserInputCfg->systemIPAddress, ".");
	k = 0;
	while (token != NULL) {
		gsEthConfigMode.au8PcIpAddr[k++] = atoi(token);
		token = strtok(NULL, ".");
	}

	gsEthConfigMode.u32ConfigPortNo = dcaUserInputCfg->DCA1000ConfigPort;
	gsEthConfigMode.u32RecordPortNo = dcaUserInputCfg->DCA1000DataPort;
	
	/* FPGA configurations */
	gsFpgaConfigMode.eDataCaptureMode = ETH_STREAM;
	gsFpgaConfigMode.eDataFormatMode  = dcaUserInputCfg->dataFormat;
	gsFpgaConfigMode.eDataXferMode= CAPTURE;
	gsFpgaConfigMode.eLogMode	= RAW_MODE;
	gsFpgaConfigMode.eLvdsMode = (dcaUserInputCfg->lvdsLaneMode == 4) ? FOUR_LANE: TWO_LANE;
	gsFpgaConfigMode.u8Timer = FPGA_CONFIG_DEFAULT_TIMER;

}

/** @fn void inlineStatsCallback(strRFDCCard_InlineProcStats inlineStats, bool bOutOfSeqSetFlag)
 * @brief This function is to handle the update of inline processing summary using callbacks
 * @param [in] inlineStats [strRFDCCard_InlineProcStats] - Status code
 * @param [in] bOutOfSeqSetFlag [bool] - If OutOfSeq occured then set flag
*/
void inlineStatsCallback(strRFDCCard_InlineProcStats inlineStats,
	bool bOutOfSeqSetFlag, UINT8 u8DataIndex)
{
#if 0 /* copied from DCA application code but disabled for this tool.*/
	if (bReady)
	{
		/** Copying global variables and trigger the event to update    */
		memcpy(&gInlineStats, &inlineStats, sizeof(strRFDCCard_InlineProcStats));
		gbOutOfSeqSetFlag = bOutOfSeqSetFlag;
		gu8DataIndex = u8DataIndex;

		osalObj_Rec.SignalEvent(&sgnInlineStsUpdateWaitEvent);
	}
#endif
}

/** @fn int dca_config(void)
 * @brief This function configures DCA1000.
 * @param [in] None
*/
int dca_config(void)
{
	UINT16 u16CmdCode = 0;
	SINT32 s32CliStatus, cnt=0;
	SINT8 s8Version[MAX_NAME_LEN];

	dcaCaptureMode = 0;
	/* API Call - Read API DLL version  */
	memset(s8Version, '\0', MAX_NAME_LEN);
    if(ReadRFDCCard_DllVersion(s8Version) == SUCCESS_STATUS)
    {
		PRINT_FUNC("version read Ok");
	}

	s32CliStatus = StatusRFDCCard_EventRegister(EthernetEventHandlercallback);

	/* API Call - Ethernet connection */
    if(u16CmdCode == CMD_CODE_CLI_ASYNC_RECORD_STOP)
    {
        s32CliStatus = ConnectRFDCCard_AsyncCommandMode(gsEthConfigMode);
    }
    else
    {
		/* this needs to be called at first time */
        s32CliStatus = ConnectRFDCCard_ConfigMode(gsEthConfigMode);
    }

	if(s32CliStatus != SUCCESS_STATUS)
	{
		printf("[ERROR] Ethernet connection failed. [error %d]", s32CliStatus);
		DisconnectRFDCCard(u16CmdCode);
		return -1;
	}

	/* Reset the FPGA, to avoid any error for last execution */
	s32CliStatus = ResetRFDCCard_FPGA();

	/** Handling command response    */
	DecodeCommandStatus(s32CliStatus, "FPGA Reset");

	/** API Call - Configure FPGA    */
	s32CliStatus = ConfigureRFDCCard_Fpga(gsFpgaConfigMode);

	/** Handling command response    */
	DecodeCommandStatus(s32CliStatus, "FPGA Configuration");

	/** API Call - Configure Record  */
	s32CliStatus = ConfigureRFDCCard_Record(gsRecConfigMode);

	/** Handling command response    */
	DecodeCommandStatus(s32CliStatus, "Configure Record");

	/** API Call - System aliveness  */
	s32CliStatus = HandshakeRFDCCard();
	DecodeCommandStatus(s32CliStatus, "Handshake FPGA");

	/* diconnect for config mode */
	DisconnectRFDCCard_ConfigMode();

	s32CliStatus = RecInlineProcStats_EventRegister(
		(INLINE_PROC_HANDLER)inlineStatsCallback);
	if (s32CliStatus != SUCCESS_STATUS)
	{
		printf("[ERROR] Registering Callback function failed (Inline status). error[%d]", s32CliStatus);
		return -1;
	}

	/** API Call - Ethernet connection                                       */
	s32CliStatus = ConnectRFDCCard_RecordMode(gsEthConfigMode);
	if (s32CliStatus != SUCCESS_STATUS)
	{
		printf("[ERROR] Ethernet connection failed. [error %d]", s32CliStatus);
		DisconnectRFDCCard_RecordMode();
		return -1;
	}

	return 0;
}

/** @fn int dcaStopRecording(void)
 * @brief This function stops the DCA1000 capture.
 * @param [in] None
*/
int dcaStopRecording(void)
{
	SINT32 s32CliStatus;
	UINT16 u16CmdCode = 0;
	
	/* if already stopped */
	if ((dcaCaptureMode == DCA1000_CAPTURE_STOP)/* || (dcaCaptureMode == DCA1000_CAPTURE_DONE) */)
	{
		return -1;
	}
	
	/* API Call - Ethernet connection */
	if (u16CmdCode == CMD_CODE_CLI_ASYNC_RECORD_STOP)
	{
		/** API Call - Stop Record    */
		s32CliStatus = StopRecordAsyncCmd();
	}
	else
	{
		s32CliStatus = StopRecordData();
	}
	
    if(s32CliStatus != SUCCESS_STATUS)
    {
        /** Handling command response */
        DecodeCommandStatus(s32CliStatus, "Stop Record Command");
    }
    else
    {
		DisconnectRFDCCard_RecordMode();
    }
	return 0;
}

/** @fn int DCA_triggerRecord(void)
 * @brief This function triggers the DCA1000 capture.
 * @param [in] None
*/
int DCA_triggerRecord(void)
{
	SINT32 s32CliStatus;

	/** API Call - Start Record     */
	s32CliStatus = StartRecordData(gsStartRecConfigMode);

	/** Handling command response   */
	DecodeCommandStatus(s32CliStatus, "Start Record");

	return s32CliStatus;
}
