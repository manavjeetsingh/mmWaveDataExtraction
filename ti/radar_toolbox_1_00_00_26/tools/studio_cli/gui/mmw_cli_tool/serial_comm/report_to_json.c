/****************************************************************************************
* FileName     : report_to_json.c
*
* Description  : This file implements JSON Report store functionalities.
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

#include <windows.h>
#include <stdio.h>
#include <share.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
//For making report directories
#include <direct.h>
#include "mmw_main.h"
#include "mmw_config.h"
#include "ti/control/mmwavelink/mmwavelink.h"

/* picked from mmwavelink as full mmwavelink is not included in thie tool */
#define SYNC_PATTERN_LEN                    (4U)

#define PATTERN_NOT_MATCHED             ((rlInt32_t)0x0)
#define SYNC_PATTERN_MATCHED            ((rlInt32_t)0x1)
/*  Protocol SYNC Patterns */
#define D2H_SYNC_PATTERN_1              (0x1234U)
#define D2H_SYNC_PATTERN_2              (0x5678U)

#define GET_ASYNC_EVENT_MSG_ID(x)		((x >> 8) & 0x00FF)
#define GET_ASYNC_EVENT_SB_ID(x) 		(x & 0x00FF)

/*string length for reading from config file*/
#define STRINGLEN			100

#define BUF_SIZE			2048
#define MAX_CHIRP_CONFIGS	512

//Signal identifier for DCBIST signal report. 
#define MMWL_APP_PSUEDO_MSG					    (0xFFU)
#define RL_RF_DCBIST_SIG_PSEUDO_REPORT			(0xFFU)
#define RL_VERSION_PSEUDO_REPORT				(0xFEU)
#define OS_DISK_BLK_SIZE					    4096
/*Print if file pointer is not NULL */
#define FRPINTF(a,b)			 if(a!=NULL) fprintf(a,b)
#define FILE_CLOSE(a)   if (a != NULL) {fclose(a); a= NULL;}

/* mmWave Sensor Device Variants */
typedef enum Mmwave_sensor_type_enum
{
	MMWAVE_SENSOR_XWR1443 = 1,

	MMWAVE_SENSOR_XWR1642,

	MMWAVE_SENSOR_XWR1843,

	MMWAVE_SENSOR_XWR6843
}Mmwave_sensor_type_e;

/*!
 * @brief
 *  Message header for reporting detection information from data path.
 *
 * @details
 *  The structure defines the message header.
 */
typedef struct MmwDemo_output_message_header_t
{
    /*! @brief   Output buffer magic word (sync word). It is initialized to  {0x1234,0x5678} */
    rlUInt32_t    magicWord;

    /*! brief   Version: : MajorNum * 2^24 + MinorNum * 2^16 + BugfixNum * 2^8 + BuildNum   */
    rlUInt32_t     version;

    /*! @brief   Total packet length including header in Bytes */
    rlUInt32_t    totalPacketLen;
    
    /*! @brief   Frame number */
    rlUInt32_t    frameNumber;

    /*! @brief   platform type */
    rlUInt32_t    platform;
    
    /*! @brief   Number of TLVs */
    rlUInt32_t    numTLVs;
} MmwDemo_output_message_header;

/**
 * @brief
 *  Message for reporting detected objects from data path.
 *
 * @details
 *  The structure defines the message body for detected objects from from data path. 
 */
typedef struct MmwDemo_output_message_tl_t
{
    /*! @brief   TLV type */
    rlUInt16_t    type;
    
    /*! @brief   Length in bytes */
    rlUInt16_t    length;

} MmwDemo_output_message_tl;

typedef struct MagicWordPattern
{
    unsigned short sync1;
    unsigned short sync2;
}MagicWordPattern_t;
/* monitor report directory name */
char *report_folder = NULL;

extern mmw_configInfo gConfigInfo;
/* recieved report count */
rcvdReportCount_t rcvRepCnt = { 0 };

/* JSON report file handers */
FILE *rlMonTempReportDataPtr = NULL;
FILE *rlMonRxGainPhRepDataPtr = NULL;
FILE *rlMonRxIfStageRepData = NULL;
FILE *rlMonTxPowRepData0Ptr = NULL;
FILE *rlMonTxPowRepData1Ptr = NULL;
FILE *rlMonTxPowRepData2Ptr = NULL;
FILE *rlMonTxBallBreakRepData0Ptr = NULL;
FILE *rlMonTxBallBreakRepData1Ptr = NULL;
FILE *rlMonTxBallBreakRepData2Ptr = NULL;
FILE *rlMonTxGainPhaMisRepDataPtr = NULL;
FILE *rlMonSynthFreqRepDataPtr = NULL;
FILE *rlMonExtAnaSigRepDataPtr = NULL;
FILE *rlMonPmclkloIntAnaSigRepDataPtr = NULL;
FILE *rlMonGpadcIntAnaSigRepDataPtr = NULL;
FILE *rlMonPllConVoltRepDataPtr = NULL;
FILE *rlMonDccClkFreqRepDataPtr = NULL;
FILE *rlDigLatentFaultReportDataPtr = NULL;
FILE *rlRfInitCompleteDataPtr = NULL;
FILE *rlRfRunTimeCalibReportDataPtr = NULL;
FILE *rlVersionReportDataPtr = NULL;


/** @fn void doShiftDWord(unsigned char buf[])
*
*   @brief Shift the WORD in the given buffer
*
*   @param[in] input buffer
*
*   @return None
*/
void doShiftDWord(unsigned char buf[])
{
    unsigned char shiftIdx;

    /* shift each byte in recevied byte array */
    for (shiftIdx = 0U; shiftIdx < 7U; shiftIdx++)
    {
        /* overwritting each data byte with next data byte of array */
        buf[shiftIdx] = buf[shiftIdx + 1U];
    }

    /* set last byte to zero */
    buf[7U] = 0U;
}

/** @fn int findByteSequence(unsigned char *rcvMsg, rlUInt32_t msgLen,
					 rlUInt32_t *byteFound)
*
*   @brief Find the byte sequence in the given buffer
*
*   @param[in] input buffer
*   @param[in] buffer length
*   @param[out] location of found pattern
*   @return PATTERN_NOT_MATCHED or SYNC_PATTERN_MATCHED
*/
int findByteSequence(unsigned char *rcvMsg, rlUInt32_t msgLen,
					 rlUInt32_t *byteFound)
{
	unsigned char count = 0;
    int retVal;
    MagicWordPattern_t recSyncPattern = {0U, 0U};
    int errVal, byteRead = 0;

    /* check for NULL pointer */
    if (rcvMsg != (unsigned char*)NULL)
    {
        /* Read 4 bytes SYNC Pattern) */
        /* if SYNC pattern has been read properly then copy it */
        (void)memcpy(&recSyncPattern, &rcvMsg[byteRead], SYNC_PATTERN_LEN);
        byteRead ++;

	    retVal = PATTERN_NOT_MATCHED;

        /* Wait for SYNC_PATTERN from the device (when mmWaveLink is running on Ext Host*/
        while (((retVal) == (rlInt32_t)PATTERN_NOT_MATCHED))
        {
            /* check if matched with SYNC pattern Host-to-device or device-to-Host */
            if (((recSyncPattern.sync1 == D2H_SYNC_PATTERN_1) &&
                (recSyncPattern.sync2 == D2H_SYNC_PATTERN_2)))
            {
                /* set to SYNC Matched flag if H2D or D2H SYNC pattern is matching
                    for big/little endian data */
                retVal = SYNC_PATTERN_MATCHED;
				*byteFound = byteRead - 1;//SYNC_PATTERN_LEN;
				break;
            }
            else
            {
				if(byteRead >= (rlUInt16_t)(msgLen/SYNC_PATTERN_LEN))
                {
                    retVal = PATTERN_NOT_MATCHED;
					break;
                }
                else
                {
                    /*  Read next 4 bytes to Low 4 bytes of buffer */
					(void)memcpy(&recSyncPattern, &rcvMsg[byteRead], SYNC_PATTERN_LEN);
					byteRead ++;//= SYNC_PATTERN_LEN;

                    /*  Shift Buffer Up for checking if the sync is shifted */
                    //doShiftDWord(syncBuf);
                    /* increment read counter */
                    count++;
                }
            }
		}
    }    
    return retVal;
}

#ifdef DBG_PRINT_EN
extern void HandleASuccessfulRead(char *inputStr, int length);
#endif
/** @fn void MMWL_getGlobalConfigStatus(rlDevGlobalCfg_t *rlDevCfgArgs)
*
*   @brief Parse the recieved buffer for monitor reports and store it to 
*			JSON file format.
*
*   @param[in] input buffer
*   @param[in] input buffer length
*
*   @return int Success - 0, Failure - Error Code
*/
int parseRcvReport(unsigned char *inputStr, int inputLen)
{
	rlUInt32_t byteFound = 0, deviceType, retVal = -1;
	MmwDemo_output_message_header hdrMsgData = {0};
	MmwDemo_output_message_tl tlvHdr = {0};
    rlUInt32_t    totalPacketLen;
    rlUInt32_t    frameNumber;
    rlUInt32_t    platform;
    rlUInt16_t    numTLVs, cnt;
	rlUInt16_t	  msgId, asyncSB, tlvLen;
	rlUInt16_t	  currByteLoc = 0;
	unsigned char	  *rcvMsg = inputStr;
	int totalByteProcessed = 0;

	/* get the Magic word first in the rcv string */
	if(findByteSequence(inputStr, inputLen, &byteFound) == SYNC_PATTERN_MATCHED)
	{
		currByteLoc = byteFound;
		rcvMsg	   += currByteLoc;
		totalByteProcessed += byteFound;

		memcpy((void*)&hdrMsgData, rcvMsg, sizeof(MmwDemo_output_message_header));
		/* update the pointer */
		rcvMsg	   += sizeof(MmwDemo_output_message_header);
		/* update how many bytes we processed */
		totalByteProcessed += sizeof(MmwDemo_output_message_header);

		totalPacketLen	= hdrMsgData.totalPacketLen;
		frameNumber		= hdrMsgData.frameNumber;
		platform		= hdrMsgData.platform;
		numTLVs			= hdrMsgData.numTLVs;

		if((platform & 0xFFFF) == 0x1843)
		{
			deviceType = MMWAVE_SENSOR_XWR1843;
		}
		else if ((platform & 0xFFFF) == 0x1642)
		{
			deviceType = MMWAVE_SENSOR_XWR1642;
		}
		else if ((platform & 0xFFFF) == 0x1443)
		{
			deviceType = MMWAVE_SENSOR_XWR1443;
		}
		else if ((platform & 0xFFFF) == 0x6843)
		{
			deviceType = MMWAVE_SENSOR_XWR6843;
		}
		else
		{
			deviceType = 0;
		}

		/* if totalpacketlen is more than we read in this interation */
		if(totalPacketLen > inputLen)
		{
			/* TODO : we need to read remaining data for current packet */
			retVal = -1;
		}

		/* process each TLV and store data to JSON */
		for(cnt = 0; cnt < numTLVs; cnt++)
		{
			memcpy((void*)&tlvHdr, rcvMsg, sizeof(MmwDemo_output_message_tl));
			/* update the pointer */
			rcvMsg	   += sizeof(MmwDemo_output_message_tl);
			/* update how many bytes we processed */
			totalByteProcessed += sizeof(MmwDemo_output_message_tl);

			tlvLen   = tlvHdr.length;
			msgId    = GET_ASYNC_EVENT_MSG_ID(tlvHdr.type);
			asyncSB  = GET_ASYNC_EVENT_SB_ID(tlvHdr.type);
	
			/* get max size of monitoring report.*/
			if(tlvLen > 100) 
			{	
#ifdef DBG_PRINT_EN
				//if TLV len is worngly read
				HandleASuccessfulRead((rcvMsg - sizeof(MmwDemo_output_message_tl)), 
					(inputLen - totalByteProcessed - sizeof(MmwDemo_output_message_tl)));
#endif
				return -1;
				break;
			}

			if((totalByteProcessed + tlvLen) > (inputLen - byteFound))
			{
				PRINT_FUNC("Partial TLV!, currTlv:[%d], numTlv:[%d]\n", cnt, numTLVs);
				break;
			}

			/* do this for each report */
			report_write(rcvMsg, msgId, asyncSB, deviceType);
			/* update the pointer */
			rcvMsg	   += tlvLen;

			/* update how many bytes we processed */
			totalByteProcessed += tlvLen;
			
			/* check if we already processed recv len but partial packet is still on the way */
			if((totalPacketLen > (inputLen - byteFound)) && ((totalByteProcessed + 8) >  (inputLen - byteFound)))
			{
				PRINT_FUNC("remaining Mon data %d \n", (totalPacketLen - (inputLen - byteFound)));
				break;
			}
		}
	}
	else
	{
		PRINT_FUNC("pattern not matched \n\r");
		/* drop this packet */
	}
}

/** @fn void send_data(unsigned char *buf, rlUInt16_t msgId, rlUInt16_t 
						asyncSB, int deviceType)
*
*   @brief Function is used as an abstraction to write monitor report to specific JSON file.
*
*   @param[in] input buffer
*   @param[in] Message ID of monitor report
*   @param[in] Sub-ID of monitor report
*   @param[in] device type
*
*   @return None
*/
void send_data(unsigned char *buf, rlUInt16_t msgId, rlUInt16_t asyncSB, int deviceType)
{
	static int ctr=0;

	switch(msgId)
	{
		case MMWL_APP_PSUEDO_MSG:
		{
			switch(asyncSB)
			{
				case RL_RF_DCBIST_SIG_PSEUDO_REPORT:
				{
					//FRPINTF(rlRfGetInternalConfigDataPtr, (const char*)buf);
					break;
				}
				
				case RL_VERSION_PSEUDO_REPORT:
				{
					FRPINTF(rlVersionReportDataPtr,(const char*)buf);
					break;
				}
					
			}
			break;
		}
		
		case RL_RF_ASYNC_EVENT_MSG:
		{
			switch(asyncSB)
			{
				case RL_RF_AE_INITCALIBSTATUS_SB:
				{
					FRPINTF(rlRfInitCompleteDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
				{
					FRPINTF(rlRfRunTimeCalibReportDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_DIG_LATENTFAULT_REPORT_SB:
				{
					FRPINTF(rlDigLatentFaultReportDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_TEMPERATURE_REPORT_SB:
				{
					FRPINTF(rlMonTempReportDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_RX_GAIN_PHASE_REPORT:
				{
					FRPINTF(rlMonRxGainPhRepDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_RX_IF_STAGE_REPORT:
				{
					FRPINTF(rlMonRxIfStageRepData,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_TX0_POWER_REPORT:
				{
					FRPINTF(rlMonTxPowRepData0Ptr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_TX1_POWER_REPORT:
				{
					FRPINTF(rlMonTxPowRepData1Ptr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_TX2_POWER_REPORT:
				{
					FRPINTF(rlMonTxPowRepData2Ptr,(const char*)buf);
					break;
				}
				
				case RL_RF_AE_MON_TX0_BALLBREAK_REPORT:
				{
					FRPINTF(rlMonTxBallBreakRepData0Ptr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_TX1_BALLBREAK_REPORT:
				{
					FRPINTF(rlMonTxBallBreakRepData1Ptr,(const char*)buf);
					break;
				}
				
				
				default:
				{
					PRINT_FUNC("unhandled asyncSB %d in send_data\n", asyncSB);
					break;
				}
			}
			break;
		}
		
		case RL_RF_ASYNC_EVENT_1_MSG:
		{
			switch(asyncSB)
			{
				
				case RL_RF_AE_MON_TX2_BALLBREAK_REPORT:
				{
					FRPINTF(rlMonTxBallBreakRepData2Ptr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT:
				{
					FRPINTF(rlMonTxGainPhaMisRepDataPtr,(const char*)buf);
					break;
				}

				case RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT:
				{
					FRPINTF(rlMonSynthFreqRepDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT:
				{
					FRPINTF(rlMonExtAnaSigRepDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT:
				{
					FRPINTF(rlMonPmclkloIntAnaSigRepDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT:
				{
					FRPINTF(rlMonGpadcIntAnaSigRepDataPtr,(const char*)buf);
					break;
				}

				case RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT:
				{
					FRPINTF(rlMonPllConVoltRepDataPtr,(const char*)buf);
					break;
				}
				case RL_RF_AE_MON_DCC_CLK_FREQ_REPORT:
				{
					FRPINTF(rlMonDccClkFreqRepDataPtr,(const char*)buf);
					break;
				}
				default:
				{
					PRINT_FUNC("unhandled asyncSB %d in send_data\n", asyncSB);
					break;
				}
			}
			break;
		}
	}
}

/** @fn int create_report_folder(char *folderPath)
*
*   @brief Create directory based on timestamp to store the monitor JSON reports.
*
*   @param[in] directory path string
*
*   @return success or failure
*/
int create_report_folder(char *folderPath)
{
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	/* create directory if not exist */
	_mkdir(folderPath);

	report_folder = (char*)malloc(STRINGLEN);
	/* get current timestamp for directory name */
	sprintf(report_folder, "%s/%d-%d-%d-%d-%d-%d", folderPath, tm.tm_year +\
		1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	PRINT_FUNC("Creating Report Folder %s\n", report_folder);
		
	/* create directory with timestamp name */
	return _mkdir(report_folder);
}

/** @fn void open_report_handles()
*
*   @brief Create Monitor JSON file and store file handles.
*			set DISK buffer for each of file handles.
*
*   @param[in] None
*
*   @return NONE
*/
void open_report_handles()
{
	char path[STRINGLEN];
	sprintf(path,"%s/%s", report_folder, "rlMonTempReportData.json");
	rlMonTempReportDataPtr = fopen(path, "w");
	setvbuf(rlMonTempReportDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonRxGainPhRepData.json");
	rlMonRxGainPhRepDataPtr = fopen(path, "w");
	setvbuf(rlMonRxGainPhRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonRxIfStageRepData.json");
	rlMonRxIfStageRepData = fopen(path, "w");
	setvbuf(rlMonRxIfStageRepData, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path, "%s/%s", report_folder, "rlMonTxPowRepData0.json");
	rlMonTxPowRepData0Ptr = fopen(path, "w");
	setvbuf(rlMonTxPowRepData0Ptr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path, "%s/%s", report_folder, "rlMonTxPowRepData1.json");
	rlMonTxPowRepData1Ptr = fopen(path, "w");
	setvbuf(rlMonTxPowRepData1Ptr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);

	sprintf(path, "%s/%s", report_folder, "rlMonTxPowRepData2.json");
	rlMonTxPowRepData2Ptr = fopen(path, "w");
	setvbuf(rlMonTxPowRepData2Ptr, NULL, _IOFBF, OS_DISK_BLK_SIZE);

	sprintf(path,"%s/%s", report_folder, "rlMonTxBallBreakRepData0.json");
	rlMonTxBallBreakRepData0Ptr = fopen(path, "w");
	setvbuf(rlMonTxBallBreakRepData0Ptr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonTxBallBreakRepData1.json");
	rlMonTxBallBreakRepData1Ptr = fopen(path, "w");
	setvbuf(rlMonTxBallBreakRepData1Ptr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);

	sprintf(path, "%s/%s", report_folder, "rlMonTxBallBreakRepData2.json");
	rlMonTxBallBreakRepData2Ptr = fopen(path, "w");
	setvbuf(rlMonTxBallBreakRepData2Ptr, NULL, _IOFBF, OS_DISK_BLK_SIZE);

	sprintf(path,"%s/%s", report_folder, "rlMonTxGainPhaMisRepData.json");
	rlMonTxGainPhaMisRepDataPtr = fopen(path, "w");
	setvbuf(rlMonTxGainPhaMisRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonSynthFreqRepData.json");
	rlMonSynthFreqRepDataPtr = fopen(path, "w");
	setvbuf(rlMonSynthFreqRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonExtAnaSigRepData.json");
	rlMonExtAnaSigRepDataPtr = fopen(path, "w");
	setvbuf(rlMonExtAnaSigRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonPmclkloIntAnaSigRepData.json");
	rlMonPmclkloIntAnaSigRepDataPtr = fopen(path, "w");
	setvbuf(rlMonPmclkloIntAnaSigRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonGpadcIntAnaSigRepData.json");
	rlMonGpadcIntAnaSigRepDataPtr = fopen(path, "w");
	setvbuf(rlMonGpadcIntAnaSigRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonPllConVoltRepData.json");
	rlMonPllConVoltRepDataPtr = fopen(path, "w");
	setvbuf(rlMonPllConVoltRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlMonDccClkFreqRepData.json");
	rlMonDccClkFreqRepDataPtr = fopen(path, "w");
	setvbuf(rlMonDccClkFreqRepDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlDigLatentFaultReportData.json");
	rlDigLatentFaultReportDataPtr = fopen(path, "w");
	setvbuf(rlDigLatentFaultReportDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlRfInitCompleteData.json");
	rlRfInitCompleteDataPtr = fopen(path, "w");
	setvbuf(rlRfInitCompleteDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlRfRunTimeCalibReportData.json");
	rlRfRunTimeCalibReportDataPtr = fopen(path, "w");
	setvbuf(rlRfRunTimeCalibReportDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
	
	sprintf(path,"%s/%s", report_folder, "rlVersionReportData.json");
	rlVersionReportDataPtr = fopen(path, "w");
	setvbuf(rlVersionReportDataPtr, NULL,  _IOFBF, OS_DISK_BLK_SIZE);
}

/** @fn void close_report_handles()
*
*   @brief Close the JSON file handles.
*
*   @param[in] none
*
*   @return NONE
*/
void close_report_handles()
{
	FILE_CLOSE(rlMonTempReportDataPtr);
	FILE_CLOSE(rlMonRxGainPhRepDataPtr);
	FILE_CLOSE(rlMonRxIfStageRepData);
	FILE_CLOSE(rlMonTxPowRepData0Ptr);
	FILE_CLOSE(rlMonTxPowRepData1Ptr);
	FILE_CLOSE(rlMonTxPowRepData2Ptr);
	FILE_CLOSE(rlMonTxBallBreakRepData0Ptr);
	FILE_CLOSE(rlMonTxBallBreakRepData1Ptr);
	FILE_CLOSE(rlMonTxBallBreakRepData2Ptr);
	FILE_CLOSE(rlMonTxGainPhaMisRepDataPtr);
	FILE_CLOSE(rlMonSynthFreqRepDataPtr);
	FILE_CLOSE(rlMonExtAnaSigRepDataPtr);
	FILE_CLOSE(rlMonPmclkloIntAnaSigRepDataPtr);
	FILE_CLOSE(rlMonGpadcIntAnaSigRepDataPtr);
	FILE_CLOSE(rlMonPllConVoltRepDataPtr);
	FILE_CLOSE(rlMonDccClkFreqRepDataPtr);
	FILE_CLOSE(rlDigLatentFaultReportDataPtr);
	FILE_CLOSE(rlRfInitCompleteDataPtr);
	FILE_CLOSE(rlRfRunTimeCalibReportDataPtr);
	FILE_CLOSE(rlVersionReportDataPtr);
}

/** @fn void write_json_headers(int deviceType)
*
*   @brief Write opening JSON blocks to Monitor Report files.
*
*   @param[in] device type.
*
*   @return NONE
*/
void write_json_headers(int deviceType)
{
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TEMPERATURE_REPORT_SB, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_RX_GAIN_PHASE_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_RX_IF_STAGE_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX0_POWER_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX1_POWER_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX2_POWER_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX0_BALLBREAK_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX1_BALLBREAK_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_RUN_TIME_CALIB_REPORT_SB, deviceType);
	
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_TX2_BALLBREAK_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT, deviceType);
	send_data("{\n\t\"data\":\n\t[\n", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_DCC_CLK_FREQ_REPORT, deviceType);
}

/** @fn void write_json_trailers(int deviceType)
*
*   @brief Write Closing JSON blocks to Monitor Report files.
*
*   @param[in] device type.
*
*   @return NONE
*/
void write_json_trailers(int deviceType)
{
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TEMPERATURE_REPORT_SB, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_RX_GAIN_PHASE_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_RX_IF_STAGE_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX0_POWER_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX1_POWER_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX2_POWER_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX0_BALLBREAK_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_MON_TX1_BALLBREAK_REPORT, deviceType);
	send_data("\n\n\t]\n}", RL_RF_ASYNC_EVENT_MSG, RL_RF_AE_RUN_TIME_CALIB_REPORT_SB, deviceType);
	
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_TX2_BALLBREAK_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT, deviceType);
	send_data("\n\t]\n}", RL_RF_ASYNC_EVENT_1_MSG, RL_RF_AE_MON_DCC_CLK_FREQ_REPORT, deviceType);
}

/** @fn void report_write(unsigned char *payload, rlUInt16_t msgId, rlUInt16_t asyncSB, int deviceType)
*
*   @brief Wrapper for writing report to disk
*
*   @param[in] monitor report data
*   @param[in] Message ID of Monitor report
*   @param[in] sub-ID of monitor report.
*   @param[in] Device Type
*
*   @return NONE
*/
void report_write(unsigned char *payload, rlUInt16_t msgId, rlUInt16_t asyncSB, int deviceType)
{
	char buf[BUF_SIZE] = "";
	int i;
					
	switch(msgId)
	{
		case MMWL_APP_PSUEDO_MSG:
		{
			switch (asyncSB)
			{
				case RL_RF_DCBIST_SIG_PSEUDO_REPORT:
				{
					//data_arr has 191 signal members to be written to json file.
					unsigned char *data_arr = (unsigned char*) payload;
					
					sprintf(buf+strlen(buf), "{\n\t\"signals\":\n\t[\n");
					
					for(i=0; i<190; i++)
					{
						sprintf(buf+strlen(buf), "%d,", data_arr[i]);
					}
					sprintf(buf+strlen(buf), "%d\n", data_arr[i]);
					

					sprintf(buf+strlen(buf), "\t]\n}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
					
				}
				case RL_VERSION_PSEUDO_REPORT:
				{
					rlVersion_t *verArgs = (rlVersion_t*) payload;
					
					sprintf(buf+strlen(buf), "{\n");
					
					sprintf(buf+strlen(buf), "\t\"RF Version\":\"%2d.%2d.%2d.%2d\",\n", 
							verArgs->rf.fwMajor, verArgs->rf.fwMinor, verArgs->rf.fwBuild, verArgs->rf.fwDebug);
							
					sprintf(buf+strlen(buf), "\t\"MSS version\":\"%2d.%2d.%2d.%2d\",\n",
							verArgs->master.fwMajor, verArgs->master.fwMinor, verArgs->master.fwBuild, verArgs->master.fwDebug);
							
					sprintf(buf+strlen(buf), "\t\"mmWaveLink version\":\"%2d.%2d.%2d.%2d\",\n", 
							verArgs->mmWaveLink.major, verArgs->mmWaveLink.minor, verArgs->mmWaveLink.build, verArgs->mmWaveLink.debug);
							
					sprintf(buf+strlen(buf), "\t\"RF Patch Version\":\"%2d.%2d.%2d.%2d\",\n", 
							verArgs->rf.patchMajor, verArgs->rf.patchMinor, ((verArgs->rf.patchBuildDebug & 0xF0) >> 4), (verArgs->rf.patchBuildDebug & 0x0F));
							
					sprintf(buf+strlen(buf), "\t\"MSS Patch version\":\"%2d.%2d.%2d.%2d\"\n", 
							verArgs->master.patchMajor, verArgs->master.patchMinor, ((verArgs->master.patchBuildDebug & 0xF0) >> 4), (verArgs->master.patchBuildDebug & 0x0F));
					
					sprintf(buf+strlen(buf), "}");


					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				
			}
			break;
		}		
		case RL_RF_ASYNC_EVENT_MSG:
		{
			switch (asyncSB)
			{
				case RL_RF_AE_INITCALIBSTATUS_SB:
				{
					rlRfInitComplete_t *data = (rlRfInitComplete_t*) payload;
					/* increment the recieve counter */
					rcvRepCnt.initCalCnt++;

					sprintf(buf+strlen(buf), "{\n");
					sprintf(buf+strlen(buf), "\t\"calibStatus\":%d,\n", data->calibStatus);
					sprintf(buf+strlen(buf), "\t\"calibUpdate\":%d,\n", data->calibUpdate);
					sprintf(buf+strlen(buf), "\t\"temperature\":%d,\n", data->temperature);
					sprintf(buf+strlen(buf), "\t\"timeStamp\":%d\n", data->timeStamp);
					sprintf(buf+strlen(buf), "}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
					
				}
				case RL_RF_AE_RUN_TIME_CALIB_REPORT_SB:
				{
					rlRfRunTimeCalibReport_t *data = (rlRfRunTimeCalibReport_t*) payload;
					//First entry, don't add comma
					if (rcvRepCnt.runTimeCnt == 0)
					{
						sprintf(buf+strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf+strlen(buf), ",\n\t\t{\n");
					}
					sprintf(buf+strlen(buf), "\t\t\t\"calibErrorFlag\":%d,\n", data->calibErrorFlag);
					sprintf(buf+strlen(buf), "\t\t\t\"calibUpdateStatus\":%d,\n", data->calibUpdateStatus);
					sprintf(buf+strlen(buf), "\t\t\t\"temperature\":%d,\n", data->temperature);
					sprintf(buf+strlen(buf), "\t\t\t\"timeStamp\":%d\n", data->timeStamp);
					sprintf(buf+strlen(buf), "\t\t}");

					/* increment the recieve counter */
					rcvRepCnt.runTimeCnt++;

					send_data(buf, msgId, asyncSB, deviceType);
					break;
					
				}
				
				case RL_RF_AE_DIG_LATENTFAULT_REPORT_SB:
				{
					rlDigLatentFaultReportData_t  *data = (rlDigLatentFaultReportData_t*) payload;
					
					sprintf(buf+strlen(buf), "{\n\t\"digMonLatentFault\":%d\n", data->digMonLatentFault);
					sprintf(buf+strlen(buf), "}");
					/* increment the recieve counter */
					rcvRepCnt.digLatentCnt ++;
					send_data(buf, msgId, asyncSB, deviceType);
					break;
					
				}
				case RL_RF_AE_MON_TEMPERATURE_REPORT_SB:
				{
					rlMonTempReportData_t* report = (rlMonTempReportData_t*) payload;
					rlInt16_t *temps;
					/* for first report don't add comma */
					if (rcvRepCnt.tempMonCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					
					sprintf(buf+strlen(buf), "\t\t\t\"tempValues\":[");
					temps = (rlInt16_t*) report->tempValues;
					for(i=0; i<9; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", temps[i]);
					}
					sprintf(buf+strlen(buf), "%d", temps[9]);
					sprintf(buf+strlen(buf), "]\n");
					sprintf(buf + strlen(buf), "\t\t}");
					
					/* increment the recieve counter */
					rcvRepCnt.tempMonCnt++;

					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				// case RL_RF_AE_MON_RX_GAIN_PHASE_REPORT:
				// {
					// PRINT_FUNC("Aync event: Rx Gain Phase Report [0x%x] \n", ((rlMonRxGainPhRep_t*)payload)->statusFlags);
					// break;
				// }
				case RL_RF_AE_MON_RX_IF_STAGE_REPORT:
				{
					int i;
					rlMonRxIfStageRep_t* report = (rlMonRxIfStageRep_t*) payload;
					rlInt8_t *hpfCutOffFreqEr = (rlInt8_t*) report->hpfCutOffFreqEr;
					rlInt8_t *lpfCutOffFreqEr = (rlInt8_t*) report->lpfCutOffStopBandAtten;
					rlInt8_t *rxIfaGainErVal = (rlInt8_t*) report->rxIfaGainErVal;
					
					/* for first report don't add comma */
					if (rcvRepCnt.rxIfStageCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}

					sprintf(buf+strlen(buf),"\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"profIndex\":%d,\n", report->profIndex);
					
					sprintf(buf+strlen(buf),  "\t\t\t\"hpfCutOffFreqEr\":[");
					for(i=0; i<7; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", hpfCutOffFreqEr[i]);
					}
					sprintf(buf+strlen(buf), "%d", hpfCutOffFreqEr[7]);
					sprintf(buf+strlen(buf),  "],\n");

					sprintf(buf+strlen(buf),  "\t\t\t\"lpfCutOffFreqEr\":[");
					for(i=0; i<7; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", lpfCutOffFreqEr[i]);
					}
					sprintf(buf+strlen(buf), "%d", lpfCutOffFreqEr[7]);
					sprintf(buf+strlen(buf),  "],\n");
					
					sprintf(buf+strlen(buf),  "\t\t\t\"rxIfaGainErVal\":[");
					for(i=0; i<7; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", rxIfaGainErVal[i]);
					}
					sprintf(buf+strlen(buf), "%d", rxIfaGainErVal[7]);
					sprintf(buf+strlen(buf),  "]\n");
					sprintf(buf + strlen(buf), "\t\t}");
					
					/* increment the recieve counter */
					rcvRepCnt.rxIfStageCnt++;

					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				case RL_RF_AE_MON_TX0_POWER_REPORT:
				case RL_RF_AE_MON_TX1_POWER_REPORT:
				case RL_RF_AE_MON_TX2_POWER_REPORT:
				{
					rlMonTxPowRep_t* report = (rlMonTxPowRep_t*) payload;
					rlInt16_t *pow_vals = (rlInt16_t*) report->txPowVal;
					
					/* for first report don't add comma */
					if (((asyncSB == RL_RF_AE_MON_TX0_POWER_REPORT) && (rcvRepCnt.tx0PwrCnt == 0)) || \
						((asyncSB == RL_RF_AE_MON_TX1_POWER_REPORT) && (rcvRepCnt.tx1PwrCnt == 0)) || \
						((asyncSB == RL_RF_AE_MON_TX2_POWER_REPORT) && (rcvRepCnt.tx2PwrCnt == 0)))
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					if (asyncSB == RL_RF_AE_MON_TX0_POWER_REPORT) rcvRepCnt.tx0PwrCnt++;
					if (asyncSB == RL_RF_AE_MON_TX1_POWER_REPORT) rcvRepCnt.tx1PwrCnt++;
					if (asyncSB == RL_RF_AE_MON_TX2_POWER_REPORT) rcvRepCnt.tx2PwrCnt++;

					sprintf(buf+strlen(buf),"\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"profIndex\":%d,\n", report->profIndex);
					sprintf(buf+strlen(buf),  "\t\t\t\"txPowVal\":[");
					
					for(i=0; i<2; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", pow_vals[i]);
					}
					sprintf(buf+strlen(buf), "%d", pow_vals[2]);
					sprintf(buf+strlen(buf),  "]\n");
					sprintf(buf + strlen(buf), "\t\t}");

					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				case RL_RF_AE_MON_TX0_BALLBREAK_REPORT:
				case RL_RF_AE_MON_TX1_BALLBREAK_REPORT:
				{
					rlMonTxBallBreakRep_t * report = (rlMonTxBallBreakRep_t *) payload;

					/* for first report don't add comma */
					if (((asyncSB == RL_RF_AE_MON_TX0_BALLBREAK_REPORT) && (rcvRepCnt.tx0BallCnt == 0)) || \
						((asyncSB == RL_RF_AE_MON_TX1_BALLBREAK_REPORT) && (rcvRepCnt.tx1BallCnt == 0)))
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					if (asyncSB == RL_RF_AE_MON_TX0_BALLBREAK_REPORT) rcvRepCnt.tx0BallCnt++;
					if (asyncSB == RL_RF_AE_MON_TX1_BALLBREAK_REPORT) rcvRepCnt.tx1BallCnt++;

					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"txReflCoefVal\":%d\n", report->txReflCoefVal);
					sprintf(buf + strlen(buf), "\t\t}");

					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				
				default:
				{
					PRINT_FUNC("Unhandled report write with asyncSB:0x%x, msgID:0x%x  \n", asyncSB, msgId);
					break;
				}	
			}
			break;
		}
		
		case RL_RF_ASYNC_EVENT_1_MSG:
		{
			switch(asyncSB)
			{
		
				case RL_RF_AE_MON_TX2_BALLBREAK_REPORT:
				{
					rlMonTxBallBreakRep_t * report = (rlMonTxBallBreakRep_t *) payload;
					/* for first report don't add comma */
					if (rcvRepCnt.tx2BallCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.tx2BallCnt++;
					
					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"txReflCoefVal\":%d\n", report->txReflCoefVal);
					sprintf(buf+strlen(buf), "\t\t}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				// case RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT:
				// {
					// PRINT_FUNC("Aync event: TX Gain Mismatch Report [0x%x] \n", ((rlMonTxGainPhaMisRep_t*)payload)->statusFlags);
					// break;
				// }

				case RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT:
				{
					rlMonSynthFreqRep_t *report = (rlMonSynthFreqRep_t*) payload;
					/* for first report don't add comma */
					if (rcvRepCnt.synthFreqCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.synthFreqCnt++;
					
					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"profIndex\":%d,\n", report->profIndex);
					sprintf(buf+strlen(buf), "\t\t\t\"maxFreqErVal\":%d,\n", report->maxFreqErVal);
					sprintf(buf+strlen(buf), "\t\t\t\"freqFailCnt\":%d\n", report->freqFailCnt);
					sprintf(buf + strlen(buf), "\t\t}");

					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				// case RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT:
				// {
					// PRINT_FUNC("Aync event: External Analog Signal Report [0x%x] \n", ((rlMonExtAnaSigRep_t*)payload)->statusFlags);
					// break;
				// }
				case RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT:
				{
					rlMonPmclkloIntAnaSigRep_t *report = (rlMonPmclkloIntAnaSigRep_t *) payload;
					/* for first report don't add comma */
					if (rcvRepCnt.pmClkCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.pmClkCnt++;
					
					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"profIndex\":%d,\n", report->profIndex);
					sprintf(buf+strlen(buf), "\t\t\t\"sync20GPower\":%d\n", report->sync20GPower);
					sprintf(buf+strlen(buf), "\t\t}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
                case RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT:
                {
					rlMonGpadcIntAnaSigRep_t * report = (rlMonGpadcIntAnaSigRep_t *) payload;
					/* for first report don't add comma */
					if (rcvRepCnt.gpadcCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.gpadcCnt++;
					
					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf+strlen(buf), "\t\t\t\"gpadcRef1Val\":%d,\n", report->gpadcRef1Val);
					sprintf(buf+strlen(buf), "\t\t\t\"gpadcRef2Val\":%d\n", report->gpadcRef2Val);
					sprintf(buf+strlen(buf), "\t\t}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				case RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT:
				{
					rlMonPllConVoltRep_t* report = (rlMonPllConVoltRep_t*) payload;
					rlInt16_t *pllContVoltVal = (rlInt16_t*) report->pllContVoltVal;
					/* for first report don't add comma */
					if (rcvRepCnt.pllCtrlCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.pllCtrlCnt++;

					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					
					sprintf(buf+strlen(buf), "\t\t\t\"pllContVoltVal\":[");
					for(i=0; i<6; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", pllContVoltVal[i]);
					}
					sprintf(buf+strlen(buf), "%d", pllContVoltVal[6]);
					sprintf(buf+strlen(buf), "]\n");
					sprintf(buf+strlen(buf), "\t\t}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				case RL_RF_AE_MON_DCC_CLK_FREQ_REPORT:
				{
					rlMonDccClkFreqRep_t *report = (rlMonDccClkFreqRep_t*) payload;
					rlInt16_t *freqMeasVal = (rlInt16_t*) report->freqMeasVal;
					/* for first report don't add comma */
					if (rcvRepCnt.dccFreqCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.dccFreqCnt++;
					
					sprintf(buf+strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					
					sprintf(buf+strlen(buf), "\t\t\t\"freqMeasVal\":[");
					for(i=0; i<5; i++)
					{
						sprintf(buf+strlen(buf), "%d, ", freqMeasVal[i]);
					}
					sprintf(buf+strlen(buf), "%d", freqMeasVal[5]);
					sprintf(buf+strlen(buf), "]\n");
					sprintf(buf+strlen(buf), "\t\t}");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				
				case RL_RF_AE_MON_RX_INT_ANALOG_SIG_REPORT:
				{
					rlMonRxIntAnaSigRep_t *report = (rlMonRxIntAnaSigRep_t*) payload;

					/* for first report don't add comma */
					if (rcvRepCnt.rxIntAnaCnt == 0)
					{
						sprintf(buf + strlen(buf), "\t\t{\n");
					}
					else
					{
						sprintf(buf + strlen(buf), ",\n\t\t{\n");
					}
					/* increment the recieve counter */
					rcvRepCnt.rxIntAnaCnt++;

					sprintf(buf + strlen(buf), "\t\t\t\"statusFlags\":%d,\n", report->statusFlags);
					sprintf(buf + strlen(buf), "\t\t\t\"profIndex\":%d,\n", report->profIndex);
					sprintf(buf+strlen(buf), "\t\t}\n");
					
					send_data(buf, msgId, asyncSB, deviceType);
					break;
				}
				default:
				{
					PRINT_FUNC("Unhandled report write with asyncSB:0x%x, msgID:0x%x  \n", asyncSB, msgId);
					break;
				}			
			
				scanf(NULL);
			}
		}
	}
}
