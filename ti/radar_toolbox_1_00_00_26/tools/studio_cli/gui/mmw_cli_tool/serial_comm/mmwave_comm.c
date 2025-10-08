/****************************************************************************************
* FileName     : mmwave_comm.c
*
* Description  : This file implements communication interface (UART/SPI) functionalities
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
/* header files */
#include <windows.h>
#include <stdio.h>
#include <share.h>
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include "serial_comm.h"
#include "mmw_config.h"
#include "mmw_main.h"
#include "Json_Utils/common.h"

#define MAX_BYTE_REPORT_READ_PER_CYCLE     (256+248)

char *CONVERTED_CFG_FILE_NAME = "mmwave_converted.cfg";

extern int create_report_folder(char *folderPath);
extern void open_report_handles();
extern void write_json_headers(int deviceType);
extern int parseRcvReport(unsigned char *inputStr, int inputLen);
extern void write_json_trailers(int deviceType);
extern void close_report_handles();

extern rlDevGlobalCfg_t rlDevGlobalCfgArgs;
mmwaveCommCfg_t gMmwaveCommCfg = {0};

/* device type info and monitor timeunit for JSON file report */
mmw_configInfo gConfigInfo;
/* this EVENT is for Monitoring JSON file is closed gracefully */
HANDLE gMonCaptureSignalEvent;
HANDLE gUartComPortHandle = NULL;
/* File pointer for config file*/
FILE *mmwave_configfPtr = NULL;

/** @fn int readCliCmdFromCfgFile(char *cliCmd)
*
*   @brief Read one line at a time from cfg format file, skip comment (starts with '%') 
*
*   @param[in] input command buffer
*
*   @return command string length
*/
int readCliCmdFromCfgFile(char *cliCmd)
{
	char *s, buff[256] = {0}, cmdName[CLI_CMD_LEN] = {0}, cmdValue[10] = {0};
	int cmdLen;
	int retVal = 0;

	/*parse the parameters by reading each line of the config file*/
	while (((s = fgets(buff, sizeof buff, mmwave_configfPtr)) != NULL))
	{
		/* Skip blank lines and comments */
		if (buff[0] == '\n' || buff[0] == '%' || buff[0] == '\r')
		{
			continue;
		}
		else
		{
			MMWL_trim(buff);
			cmdLen = strlen(buff);
			strncpy(cliCmd, buff, cmdLen);
			/* ignore blankk lines in the cfg file */
			if (strlen(buff) < 1)
			{
				retVal = 0;
				continue;
			}
			/* add NewLine Char */
			cliCmd[cmdLen++] = '\n';

			retVal = cmdLen;

		    /* Parse name/value pair from line */
			s = strtok(buff, " ");
			if (s != NULL)
			{
				strncpy(cmdName, s, CLI_CMD_LEN);
				/* store the calibration monitor config parameter */
				if (strcmp(cmdName, "calibMonCfg") == 0)
				{
					s = strtok(NULL, " ");
					if (s != NULL)
					{
						strncpy(cmdValue, s, 10);
						MMWL_trim(cmdValue);
						gConfigInfo.calMonTimeUnit = atoi(cmdValue);
					}
				}
				/* if this SensorStart Command the trigger the DCA1000 for capture */
				else if(strcmp(cmdName, "sensorStart") == 0)
				{
					mmw_TriggerDcaCapture();
				}
			}
			break;
		}
	}

	printf("%s", cliCmd);

	return retVal;
}

/** @fn int MMWL_openConfigFile(char *filePath, int cfgType)
*
*   @brief Opens MMWave config file
*
*   @param[in] input file path string
*   @param[in] configuration file type (cfg/json)
*
*   @return Success/Error
*/
int MMWL_openConfigFile(char *filePath, int cfgType)
{
	if (cfgType == MMWAVE_CONFIG_JSON_FORMAT)
	{
		mmwave_configfPtr = fopen(CONVERTED_CFG_FILE_NAME, "rb");
		if (mmwave_configfPtr == NULL)
		{
			PRINT_FUNC("failed to open config file\n");
			return -1;
		}
	}
	else
	{
		/*open config file to read parameters*/
		if ((mmwave_configfPtr == NULL) && (filePath != NULL))
		{
			mmwave_configfPtr = fopen(filePath, "rb");

			if (mmwave_configfPtr == NULL)
			{
				PRINT_FUNC("failed to open config file\n");
				return -1;
			}
		}
	}    
    return 0;
}

/** @fn void mmwaveUpdateConfig(rlDevGlobalCfg_t *devCfg)
*
*   @brief Update the global variable based on the inputs.
*
*   @param[in] pointer to global device config
*
*   @return None
*/
void mmwaveUpdateConfig(rlDevGlobalCfg_t *devCfg)
{
	if(devCfg == NULL)
		return;
	/* same COM Port is being used for sending command to device and
	 * later reading monitoring report */
	memset(gMmwaveCommCfg.comPortStr, 0, sizeof(gMmwaveCommCfg.comPortStr));
	/* update global variables */
	gMmwaveCommCfg.comPortNum = devCfg->comPortNum;
	sprintf(gMmwaveCommCfg.comPortStr, "\\\\.\\COM%d", gMmwaveCommCfg.comPortNum);

	/* store the device configuration if enabled in config.txt file */
	if (devCfg->mmWaveConfigEn == 1)
	{
		gMmwaveCommCfg.mmwCfgFormat = devCfg->mmwCfgFormat;
		strcpy(gMmwaveCommCfg.mmwConfigPath, devCfg->mmwConfigPath);
	}
	
	gMmwaveCommCfg.monitorCaptureEn = devCfg->monitorCaptureEn;
	/* store the mmwave device type */
	gConfigInfo.deviceType = devCfg->mmwaveDevVariant;

	/* store the monitor report input parameters */
	if(devCfg->monitorCaptureEn == 1)
	{
		gMmwaveCommCfg.totalReportToSave = (devCfg->totalReportToStore == 0) ? \
			~devCfg->totalReportToStore : devCfg->totalReportToStore;
		strcpy(gMmwaveCommCfg.monReportPath, devCfg->monReportPath);
		/* create a event for monitor capture purpose */
		gMonCaptureSignalEvent = CreateEvent(NULL,        // no security
			TRUE,       // manual-reset event
			FALSE,      // not signaled
			(LPTSTR)("MonCapture")); // event name

		if (gMonCaptureSignalEvent == NULL)
		{
			printf("CreateSemaphore error: %d\n", GetLastError());
		}
	}
}

/** @fn int mmwaveDeviceConfig(void)
*
*   @brief based on mmwave device type switch the Communication interface to UART or SPI
*
*   @param[in] None
*
*   @return Success or Error
*/
int mmwaveDeviceConfig(void)
{
	int retVal = -1;
	/* if JSON Input format is selected then parse that JSON config file and
	 * convert to CFG file format. */
	if (gMmwaveCommCfg.mmwCfgFormat == MMWAVE_CONFIG_JSON_FORMAT)
	{
		mmw_jsonParser(gMmwaveCommCfg.mmwConfigPath);
	}

	/* for AWR1243 and AWR2243 device SPI interface is used to
     * communicate with mmwave Sensor: Make sure that DCA1000 or DevPack board is connected with BOOST */
    if((gConfigInfo.deviceType == AWR1243) || \
       (gConfigInfo.deviceType == AWR2243))
    {
		extern mmwave_sensor_config_t  gMmwSensCfg;
		/* currently JSON is not generated with HSICClkConfig data so setting up that manually */
		gMmwSensCfg.datapathCfg.devHsiClkCfg.hsiClk = 0x9;
		printf("Configuring mmWave Sensor Device over SPI... \n");
        retVal = MMWL_App(gConfigInfo.deviceType);
    }
    else
    {	
		/* if it is JSON format then first covert that to cfg format and open converted CFG file */
		MMWL_openConfigFile(gMmwaveCommCfg.mmwConfigPath, gMmwaveCommCfg.mmwCfgFormat);
		printf("Configuring mmWave Sensor Device over UART... \n");
		gMmwaveCommCfg.commHandle = serialComm_Setup(gMmwaveCommCfg.comPortStr);
		if (gMmwaveCommCfg.commHandle == NULL)
		{
			retVal = -1;
		}
		else
		{
			gUartComPortHandle = gMmwaveCommCfg.commHandle;
			retVal = serialComm_CmdWrRd(gMmwaveCommCfg.commHandle, rlDevGlobalCfgArgs.ccsDebugEn);
		}		
    }

	return retVal;
}

/** @fn void prepareMonJsonFiles(void)
*
*   @brief This function is to prepare the Monitor JSON File format and directories 
*
*   @param[in] None
*
*   @return None
*/
void prepareMonJsonFiles(void)
{
	/* Create directory and file format for storing the monitoring repports.
	 * Each type of monitoring reports are stored to individual JSON files.
	 * It needs to do only once */
	create_report_folder(&gMmwaveCommCfg.monReportPath[0]);
	/* Open Each Monitoring JSON file for later writing data to it. */
	open_report_handles();
	/* Write the JSON header data to each monitoring JSON files */
	write_json_headers(gConfigInfo.deviceType);
}

/** @fn void mmwaveStopMonitorCapture(void)
*
*   @brief This function is to stop the Monitoring capture task 
*
*   @param[in] None
*
*   @return None
*/
void mmwaveStopMonitorCapture(void)
{
	DWORD dwWaitResult;
	/* Set Monitor capture status to active */
	rlDevGlobalCfgArgs.gModuleExecState.monCaptureState = MODULE_NOT_EXECUTING;

	dwWaitResult = WaitForSingleObject(gMonCaptureSignalEvent, 5000);
}

/** @fn int mmwaveMonitorCapture(void)
*
*   @brief This function is to Capture the monitoring report to requested path in JSON format.
*
*   @param[in] None
*
*   @return Success or Error Code
*/
int mmwaveMonitorCapture(void)
{
	char readBuff[MAX_BYTE_REPORT_READ_PER_CYCLE];
	int recvReportLength = 0;

	/* Set Monitor capture status to active */
	rlDevGlobalCfgArgs.gModuleExecState.monCaptureState = MODULE_EXECUTING;

	/* loop till we get the monitor report data from device */
	while (1)
	{
		extern mmwave_sensor_config_t  gMmwSensCfg;
		extern unsigned int gFramePeriodicity;
		extern unsigned int gFrameCount;
		extern unsigned char mmwl_bSensorStarted;
		extern unsigned int  gMonReportHdrCnt;

		unsigned int lFrameCount = gFrameCount;
		uint16_t calMonTimeUnit = gMmwSensCfg.rfCfg.calMonTimeUntCfg.calibMonTimeUnit;
		DWORD timeElapsedFromFrame;
		if (((gConfigInfo.deviceType == AWR1243) || \
			(gConfigInfo.deviceType == AWR2243)) && (gFrameTriggerTime > 0))
		{
			/* For 1243/2243: monitoring report over SPI recieved in parallel based on mmwavelink
			 * interrupt/callbacks. From MMWL_asyncEventHandler callback, it calls
			 * directly report_write function which write monitor report payload content
			 * to JSON file. */

			/* check the time elapsed from frame triggered time and check
			 * how many monitoring reports have recieved based on calMonTimeUnit */
			timeElapsedFromFrame = (GetTickCount() - gFrameTriggerTime);

			/* At Frame End Async-event stop the monitor capture process */
			if ((mmwl_bSensorStarted == 0) && (gMonReportHdrCnt >= (lFrameCount/calMonTimeUnit)))
			{
				/* sleep slightly for remaining monitor reports to recieved. */
				Sleep(gFramePeriodicity);
				gMmwaveCommCfg.totalReportToSave = 0;
			}
		}
		else
		{
			memset(readBuff, 0, MAX_BYTE_REPORT_READ_PER_CYCLE);
			/* read monitor data over serial interface */
			recvReportLength = serialComm_Read(gMmwaveCommCfg.commHandle, &readBuff[0], MAX_BYTE_REPORT_READ_PER_CYCLE);
		}
		/* if recieved non-zero length of data then parse it */
		if (recvReportLength > 0)
		{
			gMmwaveCommCfg.totalReportToSave -= 1;

			PRINT_FUNC("Monitoring Report recieved [%d]\n\r", recvReportLength);
			parseRcvReport(&readBuff[0], recvReportLength);
		}
		else
		{
			PRINT_FUNC("No Report recieved\n\r");
			//break;
		}
		/* terminate if requested via User CLI command or via DCA1000 event */
		if ((gMmwaveCommCfg.totalReportToSave == 0) || \
			(rlDevGlobalCfgArgs.gModuleExecState.monCaptureState == MODULE_NOT_EXECUTING))
		{
			break;
		}
	}
	/* apend end blocks to JSON file */
	write_json_trailers(gConfigInfo.deviceType);

	/* close all the JSON file handles */
	close_report_handles();

	/* Set EVENT to notify that JSON file is closed gracefully */
	SetEvent(gMonCaptureSignalEvent);
	
	return 0;
}
