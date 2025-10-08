/****************************************************************************************
* FileName     : mmw_config.c
*
* Description  : This file reads the mmwave configuration from config file.
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
#include <string.h>
#include "mmw_config.h"

/****************************************************************************************
* MACRO DEFINITIONS
****************************************************************************************
*/

/******************************************************************************
* GLOBAL VARIABLES/DATA-TYPES DEFINITIONS
******************************************************************************
*/

/* File pointer for config file*/
FILE *mmwl_configfPtr = NULL;

/******************************************************************************
* Function Definitions
*******************************************************************************
*/

/** @fn char *MMWL_trim(char * s)
*
*   @brief get rid of trailing and leading whitespace along with "\n"
*
*   @param[in] s - String pointer which needs to be trimed
*
*   @return int Success - 0, Failure - Error Code
*
*   get rid of trailing and leading whitespace along with "\n"
*/
char *MMWL_trim(char * s)
{
    /* Initialize start, end pointers */
    char *s1 = s, *s2 = &s[strlen(s) - 1];

    /* Trim and delimit right side */
    while ((isspace(*s2)) && (s2 >= s1))
        s2--;
    *(s2 + 1) = '\0';

    /* Trim left side */
    while ((isspace(*s1)) && (s1 < s2))
        s1++;

    /* Copy finished string */
    strcpy(s, s1);
    return s;
}


/** @fn void MMWL_getGlobalConfigStatus(rlDevGlobalCfg_t *rlDevCfgArgs)
*
*   @brief Read all global variable configurations from config file.
*
*   @param[in] rlDevGlobalCfg_t *rlDevCfgArgs
*
*   @return int Success - 0, Failure - Error Code
*/
void MMWL_getGlobalConfigStatus(rlDevGlobalCfg_t *rlDevCfgArgs)
{
	char *s, buff[256], name[STRINGLEN], value[STRINGLEN];
	int cmdStrLen = 0;
	int retVal = 0;//RL_RET_CODE_OK;
	unsigned int readAllParams = 0;
	/*seek the pointer to starting of the file so that
			we dont miss any parameter*/
	fseek(mmwl_configfPtr, 0, SEEK_SET);
	/*parse the parameters by reading each line of the config file*/
	while (((s = fgets(buff, sizeof buff, mmwl_configfPtr)) != NULL)
		&& (readAllParams == 0))
	{
		/* Skip blank lines and comments */
		if (buff[0] == '\n' || buff[0] == '#')
		{
			continue;
		}

		cmdStrLen = strlen(buff);
		/* Parse name/value pair from line */
		s = strtok(buff, "=");
		if (s == NULL)
		{
			continue;
		}
		else
		{
			strncpy(name, s, STRINGLEN);
		}
		s = strtok(NULL, "=");
		if (s == NULL)
		{
			continue;
		}
		else
		{
			strncpy(value, s, STRINGLEN);
		}
		MMWL_trim(value);
		cmdStrLen = strlen(value);

		if (strcmp(name, "MMWAVE_DEVICE_VARIANT") == 0)
		{
			strcpy(rlDevCfgArgs->mmwaveDevice, value);

			if(strcmp(value, "AWR1243") == 0)
				rlDevCfgArgs->mmwaveDevVariant = AWR1243;
			else if(strcmp(value, "AWR1443") == 0)
				rlDevCfgArgs->mmwaveDevVariant = AWR1443;
			else if(strcmp(value, "AWR1642") == 0)
				rlDevCfgArgs->mmwaveDevVariant = AWR1642;
			else if(strcmp(value, "AWR1843") == 0)
				rlDevCfgArgs->mmwaveDevVariant = AWR1843;
			else if(strcmp(value, "AWR6843") == 0)
				rlDevCfgArgs->mmwaveDevVariant = AWR6843;
			else if(strcmp(value, "AWR2243") == 0)
				rlDevCfgArgs->mmwaveDevVariant = AWR2243;
			else if(strcmp(value, "IWR1443") == 0)
				rlDevCfgArgs->mmwaveDevVariant = IWR1443;
			else if(strcmp(value, "IWR1642") == 0)
				rlDevCfgArgs->mmwaveDevVariant = IWR1642;
			else if(strcmp(value, "IWR1843") == 0)
				rlDevCfgArgs->mmwaveDevVariant = IWR1843;
			else if(strcmp(value, "IWR6843") == 0)
				rlDevCfgArgs->mmwaveDevVariant = IWR6843;
		}
		if (strcmp(name, "COM_PORT_NUM") == 0)
			rlDevCfgArgs->comPortNum = atoi(value);

		if (strcmp(name, "ENABLE_DCA_CAPTURE") == 0)
			rlDevCfgArgs->dcaCaptureEn = atoi(value);
		
		if (strcmp(name, "ENABLE_CONFIG_MMWAVE") == 0)
			rlDevCfgArgs->mmWaveConfigEn = atoi(value);

		if (strcmp(name, "ENABLE_MONITOR_CAPTURE") == 0)
			rlDevCfgArgs->monitorCaptureEn = atoi(value);

		if (strcmp(name, "ENABLE_POSTPROC") == 0)
			rlDevCfgArgs->postProcEn = atoi(value);

		if (strcmp(name, "CONFIG_FILE_FORMAT") == 0)
			rlDevCfgArgs->mmwCfgFormat = atoi(value);

		if (strcmp(name, "CONFIG_JSON_CFG_PATH") == 0)
			strcpy(&rlDevCfgArgs->mmwConfigPath[0], value);

		if (strcmp(name, "MONITORING_JSON_PATH") == 0)
			strcpy(&rlDevCfgArgs->monReportPath[0], value);
		
		if (strcmp(name, "NUMBER_OF_MONITOR_REPORT_STORE") == 0)
			rlDevCfgArgs->totalReportToStore = atoi(value);
		
		if (strcmp(name, "CAPTURED_ADC_DATA_PATH") == 0)
			strcpy(&rlDevCfgArgs->dcaConfig.adcDataPath[0], value);
		
		/* DCA1000 Configurations */
		if (strcmp(name, "DCA_FILE_PREFIX") == 0)
			strcpy(&rlDevCfgArgs->dcaConfig.filePrefix[0], value);
		
		if (strcmp(name, "DCA_MAX_REC_FILE_SIZE_MB") == 0)
			rlDevCfgArgs->dcaConfig.maxRecFileSize_MB= atoi(value);
			
		if (strcmp(name, "DCA_LVDS_LANE_MODE") == 0)
		{
			rlDevCfgArgs->dcaConfig.lvdsLaneMode = atoi(value);
			/******************************************************************/
			/* These parameters are Not supported to be set by user in this tool.
			 * DCA1000 will capture num of frames that is given in JSON or cfg file 
			 * So setting default parameters to DCAConfig
			 */
			rlDevCfgArgs->dcaConfig.packetDelay_us = 25;
			rlDevCfgArgs->dcaConfig.sequenceNumberEnable = 1; 
			strcpy(&rlDevCfgArgs->dcaConfig.DCA1000IPAddress[0], "192.168.33.180");
			rlDevCfgArgs->dcaConfig.DCA1000ConfigPort = 4096;
			rlDevCfgArgs->dcaConfig.DCA1000DataPort = 4098;
			strcpy(rlDevCfgArgs->dcaConfig.DCA1000MacAddress, "12.34.56.78.90.12");
			strcpy(&rlDevCfgArgs->dcaConfig.systemIPAddress[0], "192.168.33.30");
			rlDevCfgArgs->dcaConfig.captureStopMode = 4;
			rlDevCfgArgs->dcaConfig.bytesToCapture = 10000;
			rlDevCfgArgs->dcaConfig.durationToCapture_ms = 40000;
			rlDevCfgArgs->dcaConfig.framesToCapture = 40;
			/******************************************************************/
		}

		if (strcmp(name, "DCA_DATA_FORMAT_MODE") == 0)
			rlDevCfgArgs->dcaConfig.dataFormat = atoi(value);
			
		if (strcmp(name, "POST_PROC_EXE_PATH") == 0)
			strcpy(&rlDevCfgArgs->postProcToolPath[0], value);
			
		if (strcmp(name, "CCS_DEBUG") == 0)
			rlDevCfgArgs->ccsDebugEn = atoi(value);

		if (strcmp(name, "FW_DOWNLOAD_DISABLE") == 0)
			rlDevCfgArgs->DisableFwDownload = atoi(value);
		
		if(strcmp(name, "FLASH_CONNECTED_TO_SENSOR") == 0)
			rlDevCfgArgs->IsFlashConnected = atoi(value);
		
		if (strcmp(name, "CRC_TYPE") == 0)
			rlDevCfgArgs->clientCtx.crcType = (rlCrcType_t)atoi(value);
		
		if (strcmp(name, "ACK_TIMEOUT") == 0)
		{
			rlDevCfgArgs->clientCtx.ackTimeout = atoi(value);
			readAllParams = 1;
		}
	}
}


/** @fn int MMWL_openConfigFile()
*
*   @brief Opens MMWave config file
*
*   @return int Success - 0, Failure - Error Code
*
*   Opens MMWave config file
*/
int MMWL_openToolConfigFile(void)
{
    /*open config file to read parameters*/
    if (mmwl_configfPtr == NULL)
    {
        mmwl_configfPtr = fopen("mmwaveconfig.txt", "r");
        if (mmwl_configfPtr == NULL)
        {
            printf("[ERROR] Failed to open mmwaveconfig.txt\n");
            return -1;
        }
    }
    return 0;
}


/** @fn void MMWL_closeConfigFile()
*
*   @brief Close MMWave config file
*
*   Close MMWave config file
*/
void MMWL_closeConfigFile(void)
{
    /* Close config file */
    fclose(mmwl_configfPtr);
    mmwl_configfPtr = NULL;
}

