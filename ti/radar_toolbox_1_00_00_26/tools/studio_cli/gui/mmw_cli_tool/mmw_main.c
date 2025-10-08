/****************************************************************************************
* FileName     : mmw_main.c
*
* Description  : This file implements mmWave Studio CLI features.
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

/************************************************************************************************
 * This is CLI based tool which has following features
 * 1. This tool is currently available in exe and can be run on Windows machine only. However \
 *    source code is available open to port it to other platforms.
 * 2. Input to this tool:
 *	  a. mmwaveconfig.txt: first config file which this exe reads for
 *       i) List of tasks this exe needs to perform (config, capture, process)
 *      ii) UART COM port number over which exe will communicate with mmWave device
 *     iii) Path to mmwave config JSON/cfg file; JSON file can be generated from SensingEstimator or mmWave Studio.
 *			This tool also accepts CFG file to configure sensor, template file is available in this package.
 *      iv) Path to store monitoring reports and captured ADC data binary file.
 *       v) DCA1000 config details.
 *      vi) Post Processing exe path
 * 3. Tool's Functionalities
 *    a. Read JSON or CFG file for all the RF and monitoring configuration.
 *    b. If JSON file provided then first convert it to CFG format with compatible CLI commands in it.
 *    c. Read the original or converted CFG file for each CLI command for mmwave sensor.
 *    d. Connect to provided COM port @ 921600 baud rate and send each CLI commands to mmwave Sensor.
 *    e. Configure the DCA1000 if requested to capture ADC data in mmwaveconfig.txt
 *    f. Trigger DCA1000 capture to specified location
 *    g. Capture the monitoring report (over same COM port) in JSON format at specified location.
 *    h. After requested frame/time is over invoke PostProc tool to process the captured ADC data.
 *
 ************************************************************************************************/

/******************************************************************************
* INCLUDE FILES
******************************************************************************
*/
#include <windows.h>
#include <conio.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mmw_main.h"
#include "mmw_config.h"
#include "serial_comm.h"

/****************************************************************************************
* MACRO DEFINITIONS
****************************************************************************************
*/
#define CLI_CMD_SENSOR_STOP        0xA
#define CLI_CMD_SENSOR_START       0xB
#define CLI_CMD_DCA_STOP           0xC
#define CLI_CMD_DCA_START          0xD
#define CLI_CMD_MON_CAP_START      0xE
#define CLI_CMD_MON_CAP_STOP       0xF
#define CLI_CMD_QUIT               0xFF


/******************************************************************************
* GLOBAL VARIABLES/DATA-TYPES DEFINITIONS
******************************************************************************
*/
extern void mmwaveUpdateConfig(rlDevGlobalCfg_t *devCfg);
extern int mmwaveMonitorCapture(void);
extern int mmwaveDeviceConfig(void);
extern void LoadDcaGlobalConfigs(rlDcaConfig_t *dcaUserInputCfg);
extern int dca_config(void);
extern int dcaStopRecording(void);
extern int DCA_triggerRecord(void);
extern void MMWL_getGlobalConfigStatus(rlDevGlobalCfg_t *rlDevCfgArgs);
extern int MMWL_openToolConfigFile(void);
extern void MMWL_closeConfigFile(void);


int gCliCmdRcd = 0;
/* Global variable configurations from config file */
rlDevGlobalCfg_t rlDevGlobalCfgArgs = { 0 };
HANDLE			 gSignalEvent;
HANDLE			 gCaptureOverEvent;
UINT_PTR IDT_TIMER1 = 100;
HWND timerHandler = NULL;// 110;
volatile char terminateApp = 0;
volatile char dcaCaptureDoneEvent = 0;
/******************************************************************************
* all function definations starts here
*******************************************************************************
*/

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

/** @fn void mmw_TriggerDcaCapture(void)
*
*   @brief trigget DCA1000 ADC data capture event.
*    Don't trigger it if CLI CMD 'SensorStart' is invoked.
*
*   @return none
*/
void mmw_TriggerDcaCapture(void)
{
	if (gCliCmdRcd == CLI_CMD_SENSOR_START)
	{
		return;
	}

	if (rlDevGlobalCfgArgs.dcaCaptureEn == 1)
	{
		SetEvent(gSignalEvent);
	}
}

/* This function is being called by the DCA1000 RECORD_COMPLETE event
 * It can happen even if user gives SensorStop CMD and mmwave sensor stop
 * sending LVDS data. So in that case don't set any signal event */
void mmw_CaptureDcaDone(void)
{
	if (gCliCmdRcd == CLI_CMD_SENSOR_STOP)
	{
		return;
	}
	if ((rlDevGlobalCfgArgs.dcaCaptureEn == 1) && \
		(rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState == DCA_CAPTURE_TRIG))
	{
		SetEvent(gSignalEvent);
		IDT_TIMER1 = 0;
	}
}

/* Thread function to capture the monitoring data from mmwave sensor
 * over UART or SPI. */
DWORD WINAPI monitorCaptureFunc(void* data)
{
	/* Prepare Monitor report directory and JSON file headers  */
	prepareMonJsonFiles();

	if ((_stricmp(rlDevGlobalCfgArgs.mmwaveDevice, "AWR1243") == 0) || \
		(_stricmp(rlDevGlobalCfgArgs.mmwaveDevice, "AWR2243") == 0))
	{
		/* Don't wait if Monitor Capture command is invoked */
		if (gCliCmdRcd != CLI_CMD_MON_CAP_START)
		{
			extern unsigned char mmwl_bSensorStarted;
			/* Wait till sensorStart API called over SPI. */
			while (!mmwl_bSensorStarted);
		}
	}
	else
	{
		/* Don't wait if Monitor Capture command is invoked */
		if (gCliCmdRcd != CLI_CMD_MON_CAP_START)
		{
			/* wait till sensorStart CLI CMD is being sent to the device. */
			while (rlDevGlobalCfgArgs.gModuleExecState.mmwDevCfgState != MMW_DEV_SENSOR_START);
		}
	}

	mmwaveMonitorCapture();
	return 0;
}

/* Thread function to configure/start/stop DCA1000 */
DWORD WINAPI dcaControlFunc(void* data)
{
	DWORD dwWaitResult;
	int totalFrameDuration = 0;
	extern mmwave_sensor_config_t  gMmwSensCfg;

	/* initialize DCA1000 and configure it for later capture the data */
	LoadDcaGlobalConfigs(&rlDevGlobalCfgArgs.dcaConfig);

	/* configure DCA1000 Based on the config text file */
	if (dca_config() != 0)
	{
		printf("[ERROR] DCA1000 Configuration failed!");
		return -1;
	}
	rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CONFIGURED;

	gSignalEvent = CreateEvent(NULL,        // no security
		TRUE,       // manual-reset event
		FALSE,      // not signaled
		(LPTSTR)("SignalEvent")); // event name

	if (gSignalEvent == NULL)
	{
		PRINT_FUNC("CreateSemaphore error: %d\n", GetLastError());
		return -1;
	}

	/* if sensor configuration is requested then wait for sensorStart CLI command.
	 * Trigger DCA1000 capture just before invoking sensorStart CLI cmd.
	 */
	if (rlDevGlobalCfgArgs.mmWaveConfigEn == 1)
	{
		/* wait for sensor start CLI command */
		dwWaitResult = WaitForSingleObject(gSignalEvent, 0xFFFFFFFF);
		ResetEvent(gSignalEvent);
		switch (dwWaitResult)
		{
			// The semaphore object was signaled.
			case WAIT_OBJECT_0:
				/* start DCA1000 capture */
				if (DCA_triggerRecord() == 0)
				{
					printf("DCA1000 Capture Triggered \n");
					rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CAPTURE_TRIG;
				}
				else
				{
					printf("[ERROR] DCA1000 Trigger error!\n");
				}
				break;

			// The semaphore was nonsignaled, so a time-out occurred.
			case WAIT_TIMEOUT:
				PRINT_FUNC("Thread %d: wait timed out\n", GetCurrentThreadId());
				break;
		}
	}
	else
	{
		/* if not requested mmwave config but asking for DCA1000
		 * then trigger DCA1000 capture now.*/
		if (DCA_triggerRecord() == 0)
		{
			printf("DCA1000 Capture Triggered \n");
			rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CAPTURE_TRIG;
		}
	}

	/* wait for requested capture to over */
	dwWaitResult = WaitForSingleObject(gSignalEvent, 0xFFFFFFFF);
	ResetEvent(gSignalEvent);		

	/* stop DCA capture */
	dcaStopRecording();
	rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CAPTURE_STOP;
	dcaCaptureDoneEvent = 1;
	return 0;
}

/** @fn DWORD WINAPI readCliCommand(void* data)
*
*   @brief Wait for termination command ('q' or 'Q') by user
*
*   @return 0
*/
DWORD WINAPI readCliCommand(void* data)
{
	int retVal;
	char inputCmd[256] = { 0 };
	extern HANDLE gUartComPortHandle;

	while (1)
	{
		gets(inputCmd);
		/* sensorStart. sensorStop,
		   dcaCapture, dcaStop, quit */

		/* if 'dcaCapture' command */
		if (_stricmp(inputCmd, "dcaStart") == 0)
		{
			gCliCmdRcd = CLI_CMD_DCA_START;
			/* if DCA1000 capture is already triggered then return message */
			if ((rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState == DCA_CONFIGURED) || \
				(rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState == DCA_CAPTURE_STOP))
			{
				/* trigger DCA1000 capture */
				if (DCA_triggerRecord() != 0)
					printf("[ERROR] DCA1000 Trigger error!\n");
				rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CAPTURE_TRIG;
			}
			else if (rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState == DCA_CAPTURE_TRIG)
			{
				printf("[WARNING] DCA1000 is already triggered for Capture\n");
			}
			else
			{
				printf("[ERROR] DCA1000 is not configured, check input txt file\n");
			}
		}
		/* to Stop DCA1000 capture */
		else if (_stricmp(inputCmd, "dcaStop") == 0)
		{
			gCliCmdRcd = CLI_CMD_DCA_STOP;
			if (rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState == DCA_CAPTURE_TRIG)
			{
				/* stop the DCA1000 Recording */
				dcaStopRecording();
				dcaCaptureDoneEvent = 1;
				rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CAPTURE_STOP;
			}
			else
			{
				printf("[WARNING] DCA1000 is not yet triggered\n");
			}
		}
		else if (_stricmp(inputCmd, "sensorStart") == 0)
		{
			/* if Sensor is already triggered then don't execute this CMD */
			if (rlDevGlobalCfgArgs.gModuleExecState.mmwDevCfgState == MMW_DEV_SENSOR_START)
			{
				printf("[ERROR] Sensor is already triggered!\n");
			}
			/* for AWR1243 and AWR2243 device SPI interface is used to
			 * communicate with mmwave Sensor: Make sure that DCA1000 or DevPack board is connected with BOOST */
			else if ((rlDevGlobalCfgArgs.mmwaveDevVariant == AWR1243) || \
				(rlDevGlobalCfgArgs.mmwaveDevVariant == AWR2243))
			{
				retVal = MMWL_sensorStart(RL_DEVICE_MAP_CASCADED_1);
				if (retVal != 0)
					printf("[ERROR] SensorStart [%d]\n", retVal);
				else
					printf("Frame Triggered!\n");
			}
			else
			{
				/* invoke UART CLI Command 'sensorStart' to mwmave sensor */
				serialComm_Write(gUartComPortHandle, "sensorStart \r\n");
				/* Trigger the monitor report capture */
				if ((rlDevGlobalCfgArgs.monitorCaptureEn == 1) && \
					(rlDevGlobalCfgArgs.gModuleExecState.monCaptureState == MODULE_NOT_EXECUTING))
				{
					/* create thread for Monitoring Data capture over UART */
					//CreateThread(NULL, 0, monitorCaptureFunc, NULL, 0, NULL);
					//printf("Monitor Report Capture Started!\n");
				}
			}
			/* Set the state after triggering the frame */
			rlDevGlobalCfgArgs.gModuleExecState.mmwDevCfgState = MMW_DEV_SENSOR_START;
			gCliCmdRcd = CLI_CMD_SENSOR_START;
		}
		else if (_stricmp(inputCmd, "sensorStop") == 0)
		{
			if (rlDevGlobalCfgArgs.gModuleExecState.mmwDevCfgState == MMW_DEV_SENSOR_STOP)
			{
				printf("[ERROR] Sensor is already Stopped!\n");
			}
			/* for AWR1243 and AWR2243 device SPI interface is used to
			 * communicate with mmwave Sensor: Make sure that DCA1000 or DevPack board is connected with BOOST */
			else if ((rlDevGlobalCfgArgs.mmwaveDevVariant == AWR1243) || \
				(rlDevGlobalCfgArgs.mmwaveDevVariant == AWR2243))
			{
				retVal = MMWL_sensorStop(RL_DEVICE_MAP_CASCADED_1);
				if (retVal != 0)
					printf("[ERROR] SensorStop [%d]\n", retVal);
				else
					printf("Frame Stopped!\n");
			}
			else
			{
				/* invoke UART CLI Command 'sensorStop' to mwmave sensor */
				/* Do we need to store monitoring report? */
				if ((rlDevGlobalCfgArgs.monitorCaptureEn == 1) && \
					(rlDevGlobalCfgArgs.gModuleExecState.monCaptureState == MODULE_EXECUTING))
				{
					//mmwaveStopMonitorCapture();
					//printf("Monitor Report Capture Stopped!\n");
				}

				serialComm_Write(gUartComPortHandle, "sensorStop \r\n");
			}
			/* change the state after frame stopped */
			rlDevGlobalCfgArgs.gModuleExecState.mmwDevCfgState = MMW_DEV_SENSOR_STOP;
			gCliCmdRcd = CLI_CMD_SENSOR_STOP;
		}
		else if (_stricmp(inputCmd, "monCapStop") == 0)
		{
			gCliCmdRcd = CLI_CMD_MON_CAP_STOP;
			/* Stop the monitor report capture */
			if ((rlDevGlobalCfgArgs.monitorCaptureEn == 1) && \
				(rlDevGlobalCfgArgs.gModuleExecState.monCaptureState == MODULE_EXECUTING))
			{
				mmwaveStopMonitorCapture();
				printf("Monitor Report Capture Stopped!\n");
			}
			else
			{
				printf("[ERROR] Monitor Report Capture Already Stopped!\n");
			}
		}
		else if (_stricmp(inputCmd, "monCapStart") == 0)
		{
			gCliCmdRcd = CLI_CMD_MON_CAP_START;
			/* Do we need to store monitoring report? */
			if ((rlDevGlobalCfgArgs.monitorCaptureEn == 1) && \
				(rlDevGlobalCfgArgs.gModuleExecState.monCaptureState == MODULE_NOT_EXECUTING))
			{
				/* create thread for Monitoring Data capture over UART */
				CreateThread(NULL, 0, monitorCaptureFunc, NULL, 0, NULL);
				printf("Monitor Report Capture Started!\n");
			}
			else
			{
				printf("[ERROR] Monitor Report Capture is already running!\n");
			}
		}
		else if (_stricmp(inputCmd, "quit") == 0)
		{
			/* set TAG to terminate this app */
			terminateApp = 1;
			gCliCmdRcd = CLI_CMD_QUIT;
			break;
		}
		else
		{
			printf("[ERROR]invalid command\n");
		}
	}

	/* release all the resources and terminate the application */
	/* close COM Port and stop monitor report capture */
	if (((rlDevGlobalCfgArgs.mmWaveConfigEn == 1) || (rlDevGlobalCfgArgs.monitorCaptureEn == 1))
		&& (rlDevGlobalCfgArgs.gModuleExecState.monCaptureState == MODULE_EXECUTING))
	{
		mmwaveStopMonitorCapture();

	}
	/* if capture is ongoing then stop and terminate DCA1000 connection */
	if (rlDevGlobalCfgArgs.dcaCaptureEn == 1)
	{
		if (rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState == DCA_CAPTURE_TRIG)
		{
			/* stop the DCA1000 Recording */
			dcaStopRecording();
			dcaCaptureDoneEvent = 1;
			SetEvent(gCaptureOverEvent);
			rlDevGlobalCfgArgs.gModuleExecState.dcaCaptureState = DCA_CAPTURE_STOP;
		}
	}
	/* Close the COM Port */
	serialComm_Close(gUartComPortHandle);
	/* wait for a moment */
	Sleep(1);

	return 0;
}

/* Validate the input parameters from mmwaveoncig.txt to be in fixed combination */
int validateInputParameter()
{
	int retVal = 0;

	if ((rlDevGlobalCfgArgs.postProcEn == 1) && \
		(rlDevGlobalCfgArgs.mmWaveConfigEn == 1) && \
		(rlDevGlobalCfgArgs.dcaCaptureEn == 0))
	{
		printf("[ERROR] mmwaveconfig.txt: Enable_DCA_Capture is not set\n");
		retVal = -1;
	}
	else if ((rlDevGlobalCfgArgs.mmWaveConfigEn == 0) && \
			 (rlDevGlobalCfgArgs.dcaCaptureEn == 1))
	{
		printf("[ERROR] mmwaveconfig.txt: Enable_Config_Mmwave is not set\n");
		retVal = -1;
	}
	else if ((rlDevGlobalCfgArgs.mmWaveConfigEn == 0) && \
			 (rlDevGlobalCfgArgs.monitorCaptureEn == 1))
	{
		printf("[ERROR] mmwaveconfig.txt: Enable_Config_Mmwave is not set\n");
		retVal = -1;
	}
	else if (((rlDevGlobalCfgArgs.mmwCfgFormat == 0) && \
			 (strstr(rlDevGlobalCfgArgs.mmwConfigPath, "json") == NULL)) || \
			((rlDevGlobalCfgArgs.mmwCfgFormat == 1) && \
			(strstr(rlDevGlobalCfgArgs.mmwConfigPath, "cfg") == NULL)))
	{
		printf("[ERROR] mmwaveconfig.txt: CONFIG_JSON_CFG_PATH doesn't match with CONFIG_FILE_FORMAT\n");
		retVal = -1;
	}
	else if ((rlDevGlobalCfgArgs.dcaConfig.lvdsLaneMode == 4) && \
		(rlDevGlobalCfgArgs.mmwaveDevVariant > AWR2243))
	{
		printf("[ERROR] mmwaveconfig.txt: DCA_LVDS_LANE_MODE is not correct as per MMWAVE_DEVICE_VARIANT\n");
		retVal = -1;
	}
	return retVal;
}

/** @fn int main()
*
*   @brief Main function.
*
*   @return none
*
*   Main function.
*/
void main(void)
{
    int retVal = -1;
	HANDLE   dcaThreadHandle;
	char fullPath[_MAX_PATH];
	char postProcCmd[512] = { 0 };
	int postProcCmdLen = 0;

	/* reset global strucutre */
	memset(&rlDevGlobalCfgArgs, 0, sizeof(rlDevGlobalCfg_t));

	printf("================= mmWave Studio CLI Application ====================\n\n");

	/* open a thread to listen quit command over command prompt */
	dcaThreadHandle = CreateThread(NULL, 0, readCliCommand, NULL, 0, NULL);

	/* Open mmwaveconfig.txt file for input parameters */
	if (MMWL_openToolConfigFile() < 0)
	{
		terminateApp = 1;
		goto EXIT;
	}
	/* read mmwaveconfig.txt for all input params */
	MMWL_getGlobalConfigStatus(&rlDevGlobalCfgArgs);
	/* Close the config file after loading all the global config*/
	MMWL_closeConfigFile();

	/* validate each input parameter combination if not valid then terminate
	 * with error message */
	if (validateInputParameter() < 0)
	{
		terminateApp = 1;
		goto EXIT;
	}
	/* update global variables based on the config params */
	mmwaveUpdateConfig(&rlDevGlobalCfgArgs);
		
	/* check if DCA1000 is being requested to use to capture ADC data */
	if(rlDevGlobalCfgArgs.dcaCaptureEn == 1)
	{
		dcaThreadHandle = CreateThread(NULL, 0, dcaControlFunc, NULL, 0, NULL);
	}

	/* Do we need to store monitoring report? */
	if (rlDevGlobalCfgArgs.monitorCaptureEn == 1)
	{
		/* create thread for Monitoring Data capture over UART */
		dcaThreadHandle = CreateThread(NULL, 0, monitorCaptureFunc, NULL, 0, NULL);
	}

	/* if mmwave device configuration is being requested */
	if(rlDevGlobalCfgArgs.mmWaveConfigEn == 1)
	{
		/* @note: dca1000 capture is triggered before sending sensorStart command */

		/* read cfg or JSON file, connect over COM port or FTDI SPI.
		 * and send command to device */
		retVal = mmwaveDeviceConfig();
		if (retVal != 0)
		{
			printf("[ERROR] During Device Configuration [%d]\n", retVal);
			terminateApp = 1;
			goto EXIT;
		}
		else
		{
			/* change the state just before invoking sensorStart CMD to device */
			rlDevGlobalCfgArgs.gModuleExecState.mmwDevCfgState = MMW_DEV_SENSOR_START;
			printf("mmWave Config Done!\n");
		}
	}

	if (rlDevGlobalCfgArgs.dcaCaptureEn == 1)
	{
		/* if dca1000 capture is enable then wait for that to over then postProc 
		 * It gets RECORD_COMPLETE event at request num of frames are over.
		 */
		do {
			Sleep(1);
			SwitchToThread();
		} while (dcaCaptureDoneEvent == 0);

		/* if DCA1000 gets Frame done EVENT then terminate the monitor task as well.*/
		if (rlDevGlobalCfgArgs.monitorCaptureEn) mmwaveStopMonitorCapture();

		printf("DCA1000 Capture is done!\n");
	}

	/* if post processing of captured data is being requested */
	if(rlDevGlobalCfgArgs.postProcEn == 1)
	{
		printf("Post processing the captured data...\n");

		/* invoke Post Processing Tool with the captured ADC file path */
		/* check if given path has 'exe' postfix */
		if (strstr(&rlDevGlobalCfgArgs.postProcToolPath[0], "exe") != NULL)
		{
			char adcFileFullPath[256] = { 0 };

			postProcCmdLen = strlen(rlDevGlobalCfgArgs.postProcToolPath);
			memcpy(postProcCmd, rlDevGlobalCfgArgs.postProcToolPath, \
					postProcCmdLen);
			/* add space in the command string */
			postProcCmd[postProcCmdLen] = ' ';
			postProcCmdLen += 1;


			/*** Get ADC Data file full path which will be workspace for PostProc */
			/* check if given path is full or relative */
			if (strstr(&rlDevGlobalCfgArgs.dcaConfig.adcDataPath[0], ":") != NULL)
			{
				memcpy(adcFileFullPath, rlDevGlobalCfgArgs.dcaConfig.adcDataPath, \
					strlen(rlDevGlobalCfgArgs.dcaConfig.adcDataPath));
				/* it is full path */
				memcpy(postProcCmd + postProcCmdLen, rlDevGlobalCfgArgs.dcaConfig.adcDataPath, \
					strlen(rlDevGlobalCfgArgs.dcaConfig.adcDataPath));

				postProcCmdLen += strlen(rlDevGlobalCfgArgs.dcaConfig.adcDataPath);
			}
			else
			{
				/* it is relative Path */
				if (_fullpath(adcFileFullPath, rlDevGlobalCfgArgs.dcaConfig.adcDataPath, _MAX_PATH) != NULL)
				{
					memcpy(postProcCmd + postProcCmdLen, adcFileFullPath, \
						strlen(adcFileFullPath));

					postProcCmdLen += strlen(adcFileFullPath);
				}
			}
			/* add space in the command string */
			postProcCmd[postProcCmdLen] = ' ';
			postProcCmdLen += 1;

			/* check if given path is full or relative */
			if (strstr(&rlDevGlobalCfgArgs.mmwConfigPath[0], ":") != NULL)
			{
				/* it is full path */
				memcpy(postProcCmd+ postProcCmdLen, rlDevGlobalCfgArgs.mmwConfigPath, \
					strlen(rlDevGlobalCfgArgs.mmwConfigPath));
				
				postProcCmdLen += strlen(rlDevGlobalCfgArgs.mmwConfigPath);
			}
			else
			{
				memset(fullPath, 0, sizeof(_MAX_PATH));
				/* it is relative Path */
				if (_fullpath(fullPath, rlDevGlobalCfgArgs.mmwConfigPath, _MAX_PATH) != NULL)
				{
					memcpy(postProcCmd + postProcCmdLen, fullPath, \
						strlen(fullPath));

					postProcCmdLen += strlen(fullPath);
				}
			}
			/* add space in the command string */
			postProcCmd[postProcCmdLen] = ' ';
			postProcCmdLen += 1;

			/* Copy ADC File Path to CMD string */
			memcpy(postProcCmd + postProcCmdLen, adcFileFullPath, \
				strlen(adcFileFullPath));
			postProcCmdLen += strlen(adcFileFullPath);

			memcpy(postProcCmd + postProcCmdLen, "\\", 1);
			postProcCmdLen += 1;
			memcpy(postProcCmd + postProcCmdLen, rlDevGlobalCfgArgs.dcaConfig.filePrefix,
				strlen(rlDevGlobalCfgArgs.dcaConfig.filePrefix));
			postProcCmdLen += strlen(rlDevGlobalCfgArgs.dcaConfig.filePrefix);
			memcpy(postProcCmd + postProcCmdLen, "_Raw_0.bin",
				strlen("_Raw_0.bin"));
			postProcCmdLen += strlen("_Raw_0.bin");

			/* add space in the command string */
			postProcCmd[postProcCmdLen] = ' ';
			postProcCmdLen += 1;

			memcpy(postProcCmd + postProcCmdLen, rlDevGlobalCfgArgs.mmwaveDevice,
				strlen(rlDevGlobalCfgArgs.mmwaveDevice));
			
			printf(postProcCmd);
			system(postProcCmd);
		}
		else
		{
			printf("PostProc Tool path is not valid.\n");
			terminateApp = 1;
			goto EXIT;
		}
	}

EXIT:
    /* Wait for Quit Command */
	do {
		Sleep(10);

		if (terminateApp)
		{
			printf("=========== mmWave Studio CLI Application: Exit =========== \n\n");
			Sleep(1000); 
			break;
		}
	} while (1);
}
