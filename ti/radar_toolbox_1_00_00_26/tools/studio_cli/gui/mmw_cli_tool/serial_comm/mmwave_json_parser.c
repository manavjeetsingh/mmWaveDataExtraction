/****************************************************************************************
* FileName     : mmwave_json_parser.c
*
* Description  : This file parse the JSON file and store all the config in the pre-defined structure 
*
****************************************************************************************
* (C) Copyright 2019, Texas Instruments Incorporated. - TI web address www.ti.com
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

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include "cJSON.h"
#include "mmw_config.h"


#define START_FREQ_HZ_CONVERSION(x)         ((rlUInt32_t) (x * (1000000000 / 53.644)))
#define FREQ_SLOP_KHz_CONVERSION(x)         ((rlInt16_t) ((x*1000)/48.279))
#define SAMPLING_RATE_MHz_CONVERSION(x)  ((rlUInt16_t) (x/1000))
#define M_CM_CONVERSION(x)                 ((rlInt16_t)(x*100))
#define START_FREQ_GHZ_CONVERSION(x)     (((float)(x * 53.644))/ 1000000000)
#define FREQ_SLOP_MHz_CONVERSION(x)         ((((float)(x*48.279))/1000))
#define SAMPLING_RATE_KHz_CONVERSION(x)  ((rlUInt16_t) (x*1000))
#define CM_M_CONVERSION(x)                 ((rlInt16_t)(x/100))


cJSON *mmwave_jsonParseOp = NULL; 
mmwave_sensor_config_t  gMmwSensCfg = { 0 };

/** @fn double getValueFromJsonElem(cJSON *json_elem)
*
*   @brief This function converts JSON element to integer format.
*
*   @param[in] cJSON *json_elem : JSON element
*
*   @return converted value.
*/
double getValueFromJsonElem(cJSON *json_elem)
{
	double retVal = -1;

	if (json_elem != NULL)
	{
		if (json_elem->type == cJSON_String)
		{
			/* convert string to hex then decimal */
			if ((json_elem->valuestring[0] == '0') && \
				((json_elem->valuestring[0 + 1] == 'x') || (json_elem->valuestring[0 + 1] == 'X')))
			{
				/* It is Hex value */
				retVal = (int)strtol(json_elem->valuestring, NULL, 16);
			}
			/* if it decimal no. */
			if ((json_elem->valuestring[0] >= 48) && (json_elem->valuestring[0] <= 57))
			{
				retVal = (int)strtol(json_elem->valuestring, NULL, 16);
			}

		}
		else if (json_elem->type == cJSON_Number)
		{
			/* convert the element to double */
			retVal = (json_elem->valuedouble);
		}
	}
	return retVal;
}

/** @fn int getNumOfConfig(cJSON *json_elem)
*
*   @brief This function gets the number of configuration in the given element
*
*   @param[in] cJSON *json_elem : JSON element
*
*   @return no. of element
*/
int getNumOfConfig(cJSON *json_elem)
{
	int cnt=0;
	cJSON *elem1 = json_elem->child;
	cJSON *elem2;

	/* input is node of high level of multi config structure
	 * like: rlProfiles, rlChirps, subFrameCfg, subframeDataCfg, rlBpmChirps 
	 */
	while (elem1 != NULL)
	{
		elem2 = elem1->child;
		cnt++;
		elem1 = elem1->next;
	}
	return cnt;
}

/** @fn int mmw_monitorCfgFromJson(cJSON *json_elem)
*
*   @brief This function parse all the monitor config elements and 
*		   stores those parameters in mmwavelink monitor structures.
*
*   @param[in] cJSON *json_elem : JSON element
*
*   @return success
*/
int mmw_monitorCfgFromJson(cJSON *json_elem)
{
	cJSON *temp_elem = json_elem;
	cJSON *elem1, *elem2, *elem3;
	rlUInt16_t cfgCnt, idx, idx1;

	/* not Matched then retrun immediately */
	if (strcmp(json_elem->string, "monitoringConfig") != 0)
	{
		return -1;
	}
	else
	{
		temp_elem = temp_elem->child;

		while (temp_elem != NULL)
		{
			/* rlMonAnaEnables_t */
			if (strcmp(temp_elem->string, "rlMonAnaEnables_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.anaMonEnCfg, 0, sizeof(rlMonAnaEnables_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "enMask");
				gMmwSensCfg.monitorCfg.anaMonEnCfg.enMask = (rlUInt32_t)getValueFromJsonElem(elem1);
			#if 0
				//Not available in JSON file
				elem1 = cJSON_GetObjectItem(temp_elem, "ldoScEn");
				gMmwSensCfg.monitorCfg.anaMonEnCfg.ldoScEn = getValueFromJsonElem(elem1);
			#endif	
			}
			/* rlDigMonPeriodicConf_t */
			else if (strcmp(temp_elem->string, "rlDigMonPeriodicConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.digPeriodMonCfg, 0, sizeof(rlDigMonPeriodicConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.digPeriodMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "periodicEnableMask");
				gMmwSensCfg.monitorCfg.digPeriodMonCfg.periodicEnableMask = (rlUInt32_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlTempMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.tempMonCfg, 0, sizeof(rlTempMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.tempMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "anaTempThreshMin");
				gMmwSensCfg.monitorCfg.tempMonCfg.anaTempThreshMin = (rlInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "anaTempThreshMax");
				gMmwSensCfg.monitorCfg.tempMonCfg.anaTempThreshMax = (rlInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "digTempThreshMin");
				gMmwSensCfg.monitorCfg.tempMonCfg.digTempThreshMin = (rlInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "digTempThreshMax");
				gMmwSensCfg.monitorCfg.tempMonCfg.digTempThreshMax = (rlInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "tempDiffThresh");
				gMmwSensCfg.monitorCfg.tempMonCfg.tempDiffThresh = (rlInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRxGainPhaseMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.rxGainPhMonCfg, 0, sizeof(rlRxGainPhaseMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rfFreqBitMask");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rfFreqBitMask = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "txSel");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.txSel = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxGainAbsThresh");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainAbsThresh = (rlUInt16_t)(getValueFromJsonElem(elem1)/ 0.1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxGainMismatchErrThresh");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchErrThresh = (rlUInt16_t)(getValueFromJsonElem(elem1) / 0.1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxGainFlatnessErrThresh");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainFlatnessErrThresh = (rlUInt16_t)(getValueFromJsonElem(elem1) / 0.1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxGainPhaseMismatchErrThresh");
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainPhaseMismatchErrThresh = (rlUInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxGainMismatchOffsetVal");

				idx = 0;
				idx1 = 0;
				cfgCnt = getNumOfConfig(elem1);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}

				elem1 = elem1->child;
				while (cfgCnt)
				{//TODO Complete this based on formula Studio (ScriptOps.cs) performs 
				#if 0
					elem3 = elem1->child;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
					elem3 = elem3->next;
					gMmwSensCfg.monitorCfg.rxGainPhMonCfg.rxGainMismatchOffsetVal[idx1][idx++] = (rlUInt16_t)(getValueFromJsonElem(elem3) / 0.1);
				#endif
					idx1++;
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlTxGainPhaseMismatchMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.txGainPhMisMonCfg, 0, sizeof(rlRxGainPhaseMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rfFreqBitMask");
				gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.rfFreqBitMask = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "txEn");
				gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.txEn = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxEn");
				gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.rxEn = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "txGainMismatchThresh");
				gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.txGainMismatchThresh = (rlInt16_t)(getValueFromJsonElem(elem1));
				elem1 = cJSON_GetObjectItem(temp_elem, "txPhaseMismatchThresh");
				gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.txPhaseMismatchThresh = (rlUInt16_t)(getValueFromJsonElem(elem1));
				elem1 = cJSON_GetObjectItem(temp_elem, "txGainMismatchOffsetVal");

				idx = 0;
				idx1 = 0;
				cfgCnt = getNumOfConfig(elem1);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem3 = elem1->child;
				while (cfgCnt)
				{
					//gMmwSensCfg.monitorCfg.txGainPhMisMonCfg.txGainMismatchOffsetVal[idx++] = (rlUInt8_t)(getValueFromJsonElem(elem3) / 0.8);
					cfgCnt--;
					elem3 = elem3->next;
				}

				//TODO: complete this based on Studio JSON creation formula
			}
			else if (strcmp(temp_elem->string, "rlRxNoiseMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.rxNoiseMonCfg, 0, sizeof(rlRxNoiseMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.rxNoiseMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rfFreqBitMask");
				gMmwSensCfg.monitorCfg.rxNoiseMonCfg.rfFreqBitMask = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.rxNoiseMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "noiseThresh");
				gMmwSensCfg.monitorCfg.rxNoiseMonCfg.noiseThresh = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRxIfStageMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.rxIfStageMonCfg, 0, sizeof(rlRxIfStageMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.rxIfStageMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.rxIfStageMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "hpfCutoffErrThresh");
				gMmwSensCfg.monitorCfg.rxIfStageMonCfg.hpfCutoffErrThresh = (rlUInt16_t)getValueFromJsonElem(elem1);
#ifndef SECOND_GEN_DEVICE
				//2nd Gen: lpfCutoffBandedgeThresh, lpfCutoffStopbandThresh
				elem1 = cJSON_GetObjectItem(temp_elem, "lpfCutoffErrThresh");
				gMmwSensCfg.monitorCfg.rxIfStageMonCfg.lpfCutoffErrThresh = (rlUInt16_t)getValueFromJsonElem(elem1);
#endif
				elem1 = cJSON_GetObjectItem(temp_elem, "ifaGainErrThresh");
				gMmwSensCfg.monitorCfg.rxIfStageMonCfg.ifaGainErrThresh = (rlUInt16_t)(getValueFromJsonElem(elem1) / 0.1);
			}
			else if (strcmp(temp_elem->string, "rlSynthFreqMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.synthFreqMonCfg, 0, sizeof(rlSynthFreqMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.synthFreqMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.synthFreqMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "freqErrThresh");
				gMmwSensCfg.monitorCfg.synthFreqMonCfg.freqErrThresh = (rlUInt16_t)getValueFromJsonElem(elem1)/10;
				elem1 = cJSON_GetObjectItem(temp_elem, "monStartTime");
				gMmwSensCfg.monitorCfg.synthFreqMonCfg.monStartTime = (rlInt8_t)(getValueFromJsonElem(elem1)/ 0.2);
#ifndef SECOND_GEN_DEVICE
				//TODO 2nd Gen: monitorMode, nonLiveProfileEn
#endif
			}
			else if (strcmp(temp_elem->string, "rlExtAnaSignalsMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.extAnaSigMonCfg, 0, sizeof(rlExtAnaSignalsMonConf_t));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.extAnaSigMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "signalInpEnables");
				gMmwSensCfg.monitorCfg.extAnaSigMonCfg.signalInpEnables = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "freqErrThresh");
				gMmwSensCfg.monitorCfg.extAnaSigMonCfg.signalBuffEnables = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "signalSettlingTime");

				idx = 0;
				idx1 = 0;
				cfgCnt = getNumOfConfig(elem1);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem3 = elem1->child;
				while (cfgCnt)
				{
					gMmwSensCfg.monitorCfg.extAnaSigMonCfg.signalSettlingTime[idx++] = (rlUInt8_t)(getValueFromJsonElem(elem3) / 0.8);
					cfgCnt--;
					elem3 = elem3->next;
				}

				elem1 = cJSON_GetObjectItem(temp_elem, "signalThresh");

				idx = 0;
				idx1 = 0;
				cfgCnt = getNumOfConfig(elem1);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem3 = elem1->child;
				while (cfgCnt)
				{
					gMmwSensCfg.monitorCfg.extAnaSigMonCfg.signalThresh[idx++] = (rlUInt8_t)(getValueFromJsonElem(elem3) * (256 / 1.8));
					cfgCnt--;
					elem3 = elem3->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlAllTxIntAnaSignalsMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[0], 0, sizeof(rlTxIntAnaSignalsMonConf_t) * 3);
				memset(&gMmwSensCfg.monitorCfg.allTxIntAnaSigMonCfg, 0, sizeof(rlAllTxIntAnaSignalsMonConf_t));

				elem1 = temp_elem->child;
				if (strcmp(elem1->string, "tx0IntAnaSgnlMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "profileIndx");
					gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[0].profileIndx = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[0].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					gMmwSensCfg.monitorCfg.allTxIntAnaSigMonCfg.tx0IntAnaSgnlMonCfg = &gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[0];
				}
				elem1 = elem1->next;
				if (strcmp(elem1->string, "tx1IntAnaSgnlMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "profileIndx");
					gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[1].profileIndx = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[1].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					gMmwSensCfg.monitorCfg.allTxIntAnaSigMonCfg.tx1IntAnaSgnlMonCfg = &gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[1];
				}
				elem1 = elem1->next;
				if (strcmp(elem1->string, "tx2IntAnaSgnlMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "profileIndx");
					gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[2].profileIndx = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[2].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					gMmwSensCfg.monitorCfg.allTxIntAnaSigMonCfg.tx2IntAnaSgnlMonCfg = &gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[2];
				}
			}
			else if (strcmp(temp_elem->string, "rlAllTxPowMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.txPowerMonCfg[0], 0, sizeof(rlTxPowMonConf_t) * 3);
				memset(&gMmwSensCfg.monitorCfg.allTxPowerMonCfg, 0, sizeof(rlAllTxPowMonConf_t));

				elem1 = temp_elem->child;
				if (strcmp(elem1->string, "tx0PowrMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "profileIndx");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[0].profileIndx = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[0].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "rfFreqBitMask");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[0].rfFreqBitMask = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txPowAbsErrThresh");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[0].txPowAbsErrThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txPowFlatnessErrThresh");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[0].txPowFlatnessErrThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					/* if valid RF Freq Bit Mask is given in JSON */
					if(gMmwSensCfg.monitorCfg.txPowerMonCfg[0].rfFreqBitMask > 0)
						gMmwSensCfg.monitorCfg.allTxPowerMonCfg.tx0PowrMonCfg = &gMmwSensCfg.monitorCfg.txPowerMonCfg[0];
				}
				elem1 = elem1->next;
				if (strcmp(elem1->string, "tx1PowrMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "profileIndx");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[1].profileIndx = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[1].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "rfFreqBitMask");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[1].rfFreqBitMask = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txPowAbsErrThresh");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[1].txPowAbsErrThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txPowFlatnessErrThresh");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[1].txPowFlatnessErrThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					/* if valid RF Freq Bit Mask is given in JSON */
					if (gMmwSensCfg.monitorCfg.txPowerMonCfg[1].rfFreqBitMask > 0)
						gMmwSensCfg.monitorCfg.allTxPowerMonCfg.tx1PowrMonCfg = &gMmwSensCfg.monitorCfg.txPowerMonCfg[1];
				}
				elem1 = elem1->next;
				if (strcmp(elem1->string, "tx2PowrMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "profileIndx");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[2].profileIndx = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[2].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "rfFreqBitMask");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[2].rfFreqBitMask = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txPowAbsErrThresh");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[2].txPowAbsErrThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txPowFlatnessErrThresh");
					gMmwSensCfg.monitorCfg.txPowerMonCfg[2].txPowFlatnessErrThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					/* if valid RF Freq Bit Mask is given in JSON */
					if (gMmwSensCfg.monitorCfg.txPowerMonCfg[2].rfFreqBitMask > 0)
						gMmwSensCfg.monitorCfg.allTxPowerMonCfg.tx2PowrMonCfg = &gMmwSensCfg.monitorCfg.txPowerMonCfg[2];
				}
			}
			else if (strcmp(temp_elem->string, "rlAllTxBallBreakMonCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.txBallBreakMonCfg[0], 0, sizeof(rlTxBallbreakMonConf_t) * 3);
				memset(&gMmwSensCfg.monitorCfg.allTxBallBreakMonCfg, 0, sizeof(rlAllTxBallBreakMonCfg_t));

				elem1 = temp_elem->child;
				if (strcmp(elem1->string, "tx0BallBrkMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txBallBreakMonCfg[0].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txReflCoeffMagThresh");
					gMmwSensCfg.monitorCfg.txBallBreakMonCfg[0].txReflCoeffMagThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					gMmwSensCfg.monitorCfg.allTxBallBreakMonCfg.tx0BallBrkMonCfg = &gMmwSensCfg.monitorCfg.txBallBreakMonCfg[0];
				}
				elem1 = elem1->next;
				if (strcmp(elem1->string, "tx1BallBrkMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txBallBreakMonCfg[1].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txReflCoeffMagThresh");
					gMmwSensCfg.monitorCfg.txBallBreakMonCfg[1].txReflCoeffMagThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					gMmwSensCfg.monitorCfg.allTxBallBreakMonCfg.tx1BallBrkMonCfg = &gMmwSensCfg.monitorCfg.txBallBreakMonCfg[1];
				}
				elem1 = elem1->next;
				if (strcmp(elem1->string, "tx2BallBrkMonCfg") == 0)
				{
					elem2 = cJSON_GetObjectItem(elem1, "reportMode");
					gMmwSensCfg.monitorCfg.txBallBreakMonCfg[2].reportMode = (rlUInt8_t)getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "txReflCoeffMagThresh");
					gMmwSensCfg.monitorCfg.txBallBreakMonCfg[2].txReflCoeffMagThresh = (rlUInt16_t)getValueFromJsonElem(elem2);
#ifndef SECOND_GEN_DEVICE
					//TODO 2nd Gen: txPhShiftDacMonThresh_mV
#endif
					gMmwSensCfg.monitorCfg.allTxBallBreakMonCfg.tx2BallBrkMonCfg = &gMmwSensCfg.monitorCfg.txBallBreakMonCfg[2];
				}
			}
			else if (strcmp(temp_elem->string, "rlRxIntAnaSignalsMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlPmClkLoIntAnaSignalsMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.pmClkAnaSigMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.pmClkAnaSigMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.pmClkAnaSigMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlGpadcIntAnaSignalsMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.gpadcAnaSigMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.gpadcAnaSigMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.gpadcAnaSigMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlPllContrVoltMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.pllCtrlVoltMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.pllCtrlVoltMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.pllCtrlVoltMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "signalEnables");
				gMmwSensCfg.monitorCfg.pllCtrlVoltMonCfg.signalEnables = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlDualClkCompMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.dualClkComMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.dualClkComMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.dualClkComMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "dccPairEnables");
				gMmwSensCfg.monitorCfg.dualClkComMonCfg.dccPairEnables = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRxSatMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.rxSatMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.rxSatMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.rxSatMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "satMonSel");
				gMmwSensCfg.monitorCfg.rxSatMonCfg.satMonSel = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "primarySliceDuration");
				gMmwSensCfg.monitorCfg.rxSatMonCfg.primarySliceDuration = (rlUInt16_t)(getValueFromJsonElem(elem1)/ 0.16);
				elem1 = cJSON_GetObjectItem(temp_elem, "numSlices");
				gMmwSensCfg.monitorCfg.rxSatMonCfg.numSlices = (rlUInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxChannelMask");
				gMmwSensCfg.monitorCfg.rxSatMonCfg.rxChannelMask = (rlUInt8_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlSigImgMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.sigImgMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.sigImgMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.sigImgMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "numSlices");
				gMmwSensCfg.monitorCfg.sigImgMonCfg.numSlices = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "timeSliceNumSamples");
				gMmwSensCfg.monitorCfg.sigImgMonCfg.timeSliceNumSamples = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRxMixInPwrMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg, 0, sizeof(gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg.profileIndx = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "reportMode");
				gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg.reportMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "txEnable");
				gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg.txEnable = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "thresholds");
				gMmwSensCfg.monitorCfg.rxMixInPwrMonCfg.thresholds = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else
			{
				//Nothing
			}

			NEXT:

			if (temp_elem->child == NULL)
			{
				if (temp_elem->next == NULL)
				{
					//Terminate
					break;
				}
				else
				{
					temp_elem = temp_elem->next;
				}
			}
			else
			{
				temp_elem = temp_elem->next;
			}
		}
	}
	return 0;
}

/** @fn int mmw_dataCaptureCfgFromJson(cJSON *json_elem)
*
*   @brief This function parse all the datapath config elements and
*		   stores those parameters in mmwavelink structures.
*
*   @param[in] cJSON *json_elem : JSON element
*
*   @return success
*/
int mmw_dataCaptureCfgFromJson(cJSON *json_elem)
{
	cJSON *temp_elem = json_elem;
	cJSON *elem1;

	/* not Matched then retrun immediately */
	if (strcmp(json_elem->string, "rawDataCaptureConfig") != 0)
	{
		return -1;
	}
	else
	{
		temp_elem = temp_elem->child;

		while (temp_elem != NULL)
		{
			/* rlDevDataFmtCfg_t */
			if (strcmp(temp_elem->string, "rlDevDataFmtCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.datapathCfg.devDataFmtCfg, 0, sizeof(gMmwSensCfg.datapathCfg.devDataFmtCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "iqSwapSel");
				gMmwSensCfg.datapathCfg.devDataFmtCfg.iqSwapSel = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "chInterleave");
				gMmwSensCfg.datapathCfg.devDataFmtCfg.chInterleave = (rlUInt8_t)getValueFromJsonElem(elem1);

				gMmwSensCfg.datapathCfg.devDataFmtCfg.rxChannelEn = gMmwSensCfg.rfCfg.channelCfg.rxChannelEn;
				gMmwSensCfg.datapathCfg.devDataFmtCfg.adcBits     = gMmwSensCfg.rfCfg.adcOutCfg.fmt.b2AdcBits;
				gMmwSensCfg.datapathCfg.devDataFmtCfg.adcFmt	  = gMmwSensCfg.rfCfg.adcOutCfg.fmt.b2AdcOutFmt;			
			}
			else if (strcmp(temp_elem->string, "rlDevDataPathCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.datapathCfg.devDataPathCfg, 0, sizeof(gMmwSensCfg.datapathCfg.devDataPathCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "intfSel");
				gMmwSensCfg.datapathCfg.devDataPathCfg.intfSel = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "transferFmtPkt0");
				gMmwSensCfg.datapathCfg.devDataPathCfg.transferFmtPkt0 = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "transferFmtPkt1");
				gMmwSensCfg.datapathCfg.devDataPathCfg.transferFmtPkt1 = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "cqConfig");
				gMmwSensCfg.datapathCfg.devDataPathCfg.cqConfig = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "cq0TransSize");
				gMmwSensCfg.datapathCfg.devDataPathCfg.cq0TransSize = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "cq1TransSize");
				gMmwSensCfg.datapathCfg.devDataPathCfg.cq1TransSize = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "cq2TransSize");
				gMmwSensCfg.datapathCfg.devDataPathCfg.cq2TransSize = (rlUInt8_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlDevLaneEnable_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.datapathCfg.devLaneEnCfg, 0, sizeof(gMmwSensCfg.datapathCfg.devLaneEnCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "laneEn");
				gMmwSensCfg.datapathCfg.devLaneEnCfg.laneEn = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlDevDataPathClkCfg_t") == 0)
			{
				rlUInt16_t dataRateMapper[7] = { 900, 600, 450, 400, 300, 225, 150 };
				rlUInt16_t parsedDataRate, cnt=0;
				/* reset the config structure */
				memset(&gMmwSensCfg.datapathCfg.devDataPathClkCfg, 0, sizeof(gMmwSensCfg.datapathCfg.devDataPathClkCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "laneClkCfg");
				gMmwSensCfg.datapathCfg.devDataPathClkCfg.laneClkCfg = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "dataRate_Mbps");
				parsedDataRate = (rlUInt16_t)getValueFromJsonElem(elem1);
				for (cnt = 0; cnt < sizeof(dataRateMapper); cnt++)
				{
					if (dataRateMapper[cnt] == parsedDataRate)
					{
						break;
					}
				}
				gMmwSensCfg.datapathCfg.devDataPathClkCfg.dataRate = (rlUInt8_t)cnt;
			}
			else if (strcmp(temp_elem->string, "rlDevLvdsLaneCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.datapathCfg.devLvdsLaneCfg, 0, sizeof(gMmwSensCfg.datapathCfg.devLvdsLaneCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "laneFmtMap");
				gMmwSensCfg.datapathCfg.devLvdsLaneCfg.laneFmtMap = (rlUInt16_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "laneParamCfg");
				gMmwSensCfg.datapathCfg.devLvdsLaneCfg.laneParamCfg = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else
			{
				//Nothing
			}

			NEXT:

			if (temp_elem->child == NULL)
			{
				if (temp_elem->next == NULL)
				{
					//Terminate
					break;
				}
				else
				{
					temp_elem = temp_elem->next;
				}
			}
			else
			{
				temp_elem = temp_elem->next;
			}
		}
	}
	return 0;
}

/** @fn int mmw_rfConfigFromJson(cJSON *json_elem)
*
*   @brief This function parse all the RF config elements and
*		   stores those parameters in mmwavelink structures.
*
*   @param[in] cJSON *json_elem : JSON element
*
*   @return success
*/
int mmw_rfConfigFromJson(cJSON *json_elem)
{
	cJSON *temp_elem = json_elem;
	cJSON *elem1, *elem2, *elem3, *elem4;
	int cfgCnt = 0;

	/* not Matched then retrun immediately */
	if (strcmp(json_elem->string, "rfConfig") != 0)
	{
		return -1;
	}
	else
	{
		temp_elem = temp_elem->child;

		while (temp_elem != NULL)
		{
			/* waveformType */
			if (strcmp(temp_elem->string, "waveformType") == 0)
			{
				if (strcmp(temp_elem->valuestring, "advancedFrameChirp") == 0)
				{
					gMmwSensCfg.rfCfg.dfeOutputMode = MMW_ADV_FRAME_MODE;
				}
				else if ((strcmp(temp_elem->valuestring, "singleFrameChirp") == 0) || \
						 (strcmp(temp_elem->valuestring, "legacyFrameChirp") == 0))
				{
					gMmwSensCfg.rfCfg.dfeOutputMode = MMW_LEFACY_FRAME_MODE;
				}
				else
				{	/* continuousWave */
					gMmwSensCfg.rfCfg.dfeOutputMode = MMW_CONTINUOUS_MODE;
				}
			}
			/* MIMOScheme */
			else if (strcmp(temp_elem->string, "MIMOScheme") == 0)
			{
				temp_elem->valuestring;
			}
			/* MIMOScheme */
			else if (strcmp(temp_elem->string, "rlCalibrationDataFile") == 0)
			{
				/* Not supported in this tool version, 
				 * add calibration file path */
			}
			/* channelConfig */
			else if (strcmp(temp_elem->string, "rlChanCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.channelCfg, 0, sizeof(gMmwSensCfg.rfCfg.channelCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "rxChannelEn");
				gMmwSensCfg.rfCfg.channelCfg.rxChannelEn = (rlUInt16_t)getValueFromJsonElem(elem1);

				elem1 = cJSON_GetObjectItem(temp_elem, "txChannelEn");
				gMmwSensCfg.rfCfg.channelCfg.txChannelEn = (rlUInt16_t)getValueFromJsonElem(elem1);

				elem1 = cJSON_GetObjectItem(temp_elem, "cascading");
				gMmwSensCfg.rfCfg.channelCfg.cascading = (rlUInt16_t)getValueFromJsonElem(elem1);
				/* This tool supports single device only, skipping rlChanCfg_t->cascadingPinoutCfg for now */
			}
			/* rlAdcOutCfg_t */
			else if (strcmp(temp_elem->string, "rlAdcOutCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.adcOutCfg, 0, sizeof(gMmwSensCfg.rfCfg.adcOutCfg));

				elem1 = temp_elem->child;
				if (strcmp(elem1->string, "fmt") == 0)
				{
					elem3 = cJSON_GetObjectItem(elem1, "b2AdcBits");
					gMmwSensCfg.rfCfg.adcOutCfg.fmt.b2AdcBits = (rlUInt32_t)getValueFromJsonElem(elem3);

					elem3 = cJSON_GetObjectItem(elem1, "b8FullScaleReducFctr");
					gMmwSensCfg.rfCfg.adcOutCfg.fmt.b8FullScaleReducFctr = (rlUInt32_t)getValueFromJsonElem(elem3);

					elem3 = cJSON_GetObjectItem(elem1, "b2AdcOutFmt");
					gMmwSensCfg.rfCfg.adcOutCfg.fmt.b2AdcOutFmt = (rlUInt32_t)getValueFromJsonElem(elem3);
				}
			}
			else if (strcmp(temp_elem->string, "rlLowPowerModeCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.lowPowerCfg, 0, sizeof(gMmwSensCfg.rfCfg.lowPowerCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "lpAdcMode");
				gMmwSensCfg.rfCfg.lowPowerCfg.lpAdcMode = (rlUInt16_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlProfiles") == 0)
			{
				rlProfileCfg_t profileCfg = { 0 };
				rlProfileCfg_t *profileCfgPtr = NULL;

				cfgCnt = getNumOfConfig(temp_elem);
				if (cfgCnt == 0)
				{
					gMmwSensCfg.rfCfg.profileCfgCnt = cfgCnt;
					goto NEXT;
				}
				else
				{
					gMmwSensCfg.rfCfg.profileCfgCnt = cfgCnt;
					gMmwSensCfg.rfCfg.profileCfg = (rlProfileCfg_t*)calloc(cfgCnt, sizeof(rlProfileCfg_t));
					profileCfgPtr = gMmwSensCfg.rfCfg.profileCfg;
				}
				
				elem1 = temp_elem->child;
				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "profileId");
					profileCfg.profileId = (rlUInt16_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "pfVcoSelect");
					profileCfg.pfVcoSelect = (rlUInt8_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "pfCalLutUpdate");
					profileCfg.pfCalLutUpdate = (rlUInt8_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "startFreqConst_GHz");
					profileCfg.startFreqConst = START_FREQ_HZ_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "idleTimeConst_usec");
					profileCfg.idleTimeConst = (rlUInt32_t)(getValueFromJsonElem(elem3) * 100);
					elem3 = cJSON_GetObjectItem(elem2, "adcStartTimeConst_usec");
					profileCfg.adcStartTimeConst = (rlUInt32_t)(getValueFromJsonElem(elem3) * 100);
					elem3 = cJSON_GetObjectItem(elem2, "rampEndTime_usec");
					profileCfg.rampEndTime = (rlUInt32_t)(getValueFromJsonElem(elem3) * 100);
					elem3 = cJSON_GetObjectItem(elem2, "txOutPowerBackoffCode");
					profileCfg.txOutPowerBackoffCode = (rlUInt32_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "txPhaseShifter");
					profileCfg.txPhaseShifter = (rlUInt32_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "freqSlopeConst_MHz_usec");
					profileCfg.freqSlopeConst = FREQ_SLOP_KHz_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "txStartTime_usec");
					profileCfg.txStartTime = (rlUInt16_t)(getValueFromJsonElem(elem3) * 100);
					elem3 = cJSON_GetObjectItem(elem2, "numAdcSamples");
					profileCfg.numAdcSamples = (rlUInt16_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "digOutSampleRate");
					profileCfg.digOutSampleRate = (rlUInt16_t)(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "hpfCornerFreq1");
					profileCfg.hpfCornerFreq1 = (rlUInt8_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "hpfCornerFreq2");
					profileCfg.hpfCornerFreq2 = (rlUInt8_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "rxGain_dB");
					profileCfg.rxGain = (rlUInt16_t)getValueFromJsonElem(elem3);

					memcpy(profileCfgPtr, &profileCfg, sizeof(rlProfileCfg_t));
					memset(&profileCfg, 0, sizeof(rlProfileCfg_t));
					profileCfgPtr++;
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlChirps") == 0)
			{
				rlChirpCfg_t chirpCfg = { 0 };
				rlChirpCfg_t *chirpCfgPtr = NULL;

				cfgCnt = getNumOfConfig(temp_elem);
				if (cfgCnt == 0)
				{
					gMmwSensCfg.rfCfg.chirpCfgCnt = cfgCnt;
					goto NEXT;
				}
				else
				{
					gMmwSensCfg.rfCfg.chirpCfgCnt = cfgCnt;
					gMmwSensCfg.rfCfg.chirpCfg = (rlChirpCfg_t*)calloc(cfgCnt, sizeof(rlChirpCfg_t));
					chirpCfgPtr = gMmwSensCfg.rfCfg.chirpCfg;
				}

				elem1 = temp_elem->child;
				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "chirpStartIdx");
					chirpCfg.chirpStartIdx = (rlUInt16_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "chirpEndIdx");
					chirpCfg.chirpEndIdx = (rlUInt16_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "profileId");
					chirpCfg.profileId = (rlUInt16_t)getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "startFreqVar_MHz");
					chirpCfg.startFreqVar = START_FREQ_HZ_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "freqSlopeVar_KHz_usec");
					chirpCfg.freqSlopeVar = FREQ_SLOP_KHz_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "idleTimeVar_usec");
					chirpCfg.idleTimeVar = (rlUInt16_t)(getValueFromJsonElem(elem3) * 100);
					elem3 = cJSON_GetObjectItem(elem2, "adcStartTimeVar_usec");
					chirpCfg.adcStartTimeVar = (rlUInt16_t)(getValueFromJsonElem(elem3) * 100);
					elem3 = cJSON_GetObjectItem(elem2, "txEnable");
					chirpCfg.txEnable = (rlUInt16_t)getValueFromJsonElem(elem3);

					memcpy(chirpCfgPtr, &chirpCfg, sizeof(rlChirpCfg_t));
					memset(&chirpCfg, 0, sizeof(rlChirpCfg_t));
					chirpCfgPtr++;
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlRfCalMonTimeUntConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.calMonTimeUntCfg, 0, sizeof(gMmwSensCfg.rfCfg.calMonTimeUntCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "calibMonTimeUnit");
				gMmwSensCfg.rfCfg.calMonTimeUntCfg.calibMonTimeUnit = (rlUInt16_t)getValueFromJsonElem(elem1);
				/* for single device it needs to be set to 1 */
				gMmwSensCfg.rfCfg.calMonTimeUntCfg.numOfCascadeDev = 1;
			}
			else if (strcmp(temp_elem->string, "rlRfCalMonFreqLimitConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.rfCalMonFreqLimitCfg, 0, sizeof(gMmwSensCfg.rfCfg.rfCalMonFreqLimitCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitLow_GHz");
				gMmwSensCfg.rfCfg.rfCalMonFreqLimitCfg.freqLimitLow = (rlUInt16_t)(getValueFromJsonElem(elem1)*10);
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitHigh_GHz");
				gMmwSensCfg.rfCfg.rfCalMonFreqLimitCfg.freqLimitHigh = (rlUInt16_t)(getValueFromJsonElem(elem1) * 10);
			}
			else if (strcmp(temp_elem->string, "rlRfApllSynthBWCtlConf_t") == 0)
			{
				/* Not available for 1st Gen mmwave sensor devices */			
			}
			else if (strcmp(temp_elem->string, "rlRfInitCalConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.rfInitCalCfg, 0, sizeof(gMmwSensCfg.rfCfg.rfInitCalCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "calibEnMask");
				gMmwSensCfg.rfCfg.rfInitCalCfg.calibEnMask = (rlUInt32_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfInitCalConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.rfInitCalCfg, 0, sizeof(gMmwSensCfg.rfCfg.rfInitCalCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "calibEnMask");
				gMmwSensCfg.rfCfg.rfInitCalCfg.calibEnMask = (rlUInt32_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRunTimeCalibConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.runTimeCalCfg, 0, sizeof(gMmwSensCfg.rfCfg.runTimeCalCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "oneTimeCalibEnMask");
				gMmwSensCfg.rfCfg.runTimeCalCfg.oneTimeCalibEnMask = (rlUInt32_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "periodicCalibEnMask");
				gMmwSensCfg.rfCfg.runTimeCalCfg.periodicCalibEnMask = (rlUInt32_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "calibPeriodicity");
				gMmwSensCfg.rfCfg.runTimeCalCfg.calibPeriodicity = (rlUInt32_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "reportEn");
				gMmwSensCfg.rfCfg.runTimeCalCfg.reportEn = (rlUInt8_t)getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "txPowerCalMode");
				gMmwSensCfg.rfCfg.runTimeCalCfg.txPowerCalMode = (rlUInt8_t)getValueFromJsonElem(elem1);
				/* For 2nd Gen Only: calTempIdxOverrideEnMask, TxCalTempIdx,RxCalTempIdx, LODistCalTempIdx*/
			}
			else if (strcmp(temp_elem->string, "rlFrameCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.frameCfg, 0, sizeof(gMmwSensCfg.rfCfg.frameCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "chirpEndIdx");
				gMmwSensCfg.rfCfg.frameCfg.chirpEndIdx = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "chirpStartIdx");
				gMmwSensCfg.rfCfg.frameCfg.chirpStartIdx = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "numLoops");
				gMmwSensCfg.rfCfg.frameCfg.numLoops = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "numFrames");
				gMmwSensCfg.rfCfg.frameCfg.numFrames = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "framePeriodicity_msec");
				gMmwSensCfg.rfCfg.frameCfg.framePeriodicity = getValueFromJsonElem(elem1)* (1000000 / 5);
				elem1 = cJSON_GetObjectItem(temp_elem, "triggerSelect");
				gMmwSensCfg.rfCfg.frameCfg.triggerSelect = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "frameTriggerDelay");
				gMmwSensCfg.rfCfg.frameCfg.frameTriggerDelay = getValueFromJsonElem(elem1)* (1000000 / 5);

				gMmwSensCfg.rfCfg.frameCfg.numAdcSamples = gMmwSensCfg.rfCfg.profileCfg[\
					gMmwSensCfg.rfCfg.chirpCfg[gMmwSensCfg.rfCfg.frameCfg.chirpEndIdx].profileId].numAdcSamples * 2;
			}
			else if (strcmp(temp_elem->string, "rlAdvFrameCfg_t") == 0)
			{
				unsigned char subFrmCnt = 0;
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.advFrameCfg, 0, sizeof(gMmwSensCfg.rfCfg.advFrameCfg));

				elem1 = temp_elem->child;
				if (strcmp(elem1->string, "frameSeq") == 0)
				{					
					elem2 = cJSON_GetObjectItem(elem1, "numOfSubFrames");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.numOfSubFrames = getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "forceProfile");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.forceProfile = getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "loopBackCfg");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.loopBackCfg = getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "subFrameTrigger");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameTrigger = getValueFromJsonElem(elem2);
					
					elem2 = cJSON_GetObjectItem(elem1, "subFrameCfg");
					cfgCnt = getNumOfConfig(elem2);

					elem4 = elem2->child;
					while (cfgCnt)
					{
						elem2 = elem4->child;
						elem3 = cJSON_GetObjectItem(elem2, "forceProfileIdx");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].forceProfileIdx = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "chirpStartIdx");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].chirpStartIdx = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "numOfChirps");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numOfChirps = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "numLoops");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numLoops = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "burstPeriodicity_msec");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].burstPeriodicity = getValueFromJsonElem(elem3)* (1000000/5);
						elem3 = cJSON_GetObjectItem(elem2, "chirpStartIdxOffset");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].chirpStartIdxOffset = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "numOfBurst");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numOfBurst = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "numOfBurstLoops");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numOfBurstLoops = getValueFromJsonElem(elem3);
						elem3 = cJSON_GetObjectItem(elem2, "subFramePeriodicity_msec");
						gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].subFramePeriodicity = getValueFromJsonElem(elem3)* (1000000 / 5);
						
						subFrmCnt++;
						cfgCnt--;

						elem4 = elem4->next;
					}

					elem2 = cJSON_GetObjectItem(elem1, "numFrames");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.numFrames = getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "triggerSelect");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.triggerSelect = getValueFromJsonElem(elem2);
					elem2 = cJSON_GetObjectItem(elem1, "frameTrigDelay_usec");
					gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.frameTrigDelay = getValueFromJsonElem(elem2)* (1000000 / 5);

				}

				elem2 = elem1->next;
				if (strcmp(elem2->string, "frameData") == 0)
				{
					subFrmCnt = 0;
					elem3 = cJSON_GetObjectItem(elem2, "numSubFrames");
					//TODO : Bug in JSON generation by Studio, it doesn't fill frameData->numSubFrames parameter
					gMmwSensCfg.rfCfg.advFrameCfg.frameData.numSubFrames = gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.numOfSubFrames;
					elem3 = cJSON_GetObjectItem(elem2, "subframeDataCfg");
					cfgCnt = getNumOfConfig(elem3);

					elem1 = elem3->child;
					while (cfgCnt)
					{
						elem2 = elem1->child;
						elem3 = cJSON_GetObjectItem(elem2, "totalChirps");
						gMmwSensCfg.rfCfg.advFrameCfg.frameData.subframeDataCfg[subFrmCnt].totalChirps = \
							(gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numOfChirps *
								gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numLoops *
								gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numOfBurst *
								gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].numOfBurstLoops);
						elem3 = cJSON_GetObjectItem(elem2, "numAdcSamples");
						if (gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.forceProfile)
						{
							gMmwSensCfg.rfCfg.advFrameCfg.frameData.subframeDataCfg[subFrmCnt].numAdcSamples = \
								gMmwSensCfg.rfCfg.profileCfg[gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].forceProfileIdx].numAdcSamples * 2;
						}
						else
						{
							gMmwSensCfg.rfCfg.advFrameCfg.frameData.subframeDataCfg[subFrmCnt].numAdcSamples = \
								gMmwSensCfg.rfCfg.profileCfg[gMmwSensCfg.rfCfg.chirpCfg[gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[subFrmCnt].chirpStartIdx].profileId].numAdcSamples * 2;

						}
							
						elem3 = cJSON_GetObjectItem(elem2, "numChirpsInDataPacket");
						gMmwSensCfg.rfCfg.advFrameCfg.frameData.subframeDataCfg[subFrmCnt].numChirpsInDataPacket = getValueFromJsonElem(elem3);

						subFrmCnt++;
						cfgCnt--;

						elem1 = elem1->next;
					}
				}
			}
			else if (strcmp(temp_elem->string, "rlContModeCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.contModeCfg, 0, sizeof(gMmwSensCfg.rfCfg.contModeCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "startFreqConst_GHz");
				gMmwSensCfg.rfCfg.contModeCfg.startFreqConst = START_FREQ_HZ_CONVERSION(getValueFromJsonElem(elem1));
				elem1 = cJSON_GetObjectItem(temp_elem, "txOutPowerBackoffCode");
				gMmwSensCfg.rfCfg.contModeCfg.txOutPowerBackoffCode = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "txPhaseShifter");
				gMmwSensCfg.rfCfg.contModeCfg.txPhaseShifter = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "digOutSampleRate");
				gMmwSensCfg.rfCfg.contModeCfg.digOutSampleRate = (rlUInt16_t)(getValueFromJsonElem(elem1));
				elem1 = cJSON_GetObjectItem(temp_elem, "hpfCornerFreq1");
				gMmwSensCfg.rfCfg.contModeCfg.hpfCornerFreq1 = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "hpfCornerFreq2");
				gMmwSensCfg.rfCfg.contModeCfg.hpfCornerFreq2 = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "rxGain_dB");
				gMmwSensCfg.rfCfg.contModeCfg.rxGain = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "vcoSelect");
				gMmwSensCfg.rfCfg.contModeCfg.vcoSelect = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlContModeEn_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.contModeEn, 0, sizeof(gMmwSensCfg.rfCfg.contModeEn));
				elem1 = cJSON_GetObjectItem(temp_elem, "contModeEn");
				gMmwSensCfg.rfCfg.contModeEn.contModeEn = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlBpmChirps") == 0)
			{
				rlBpmChirpCfg_t bpmChpCfg = { 0 };
				rlBpmChirpCfg_t *bpmChpCfgPtr = NULL;

				cfgCnt = getNumOfConfig(temp_elem);
				if (cfgCnt == 0)
				{
					gMmwSensCfg.rfCfg.bpmCfgCnt = cfgCnt;
					goto NEXT;
				}
				else
				{
					gMmwSensCfg.rfCfg.bpmCfgCnt = cfgCnt;
					gMmwSensCfg.rfCfg.bpmChirpCfg = (rlBpmChirpCfg_t*)calloc(cfgCnt, sizeof(rlBpmChirpCfg_t));
					bpmChpCfgPtr = gMmwSensCfg.rfCfg.bpmChirpCfg;
				}
				elem1 = temp_elem->child;
				if (strcmp(elem1->child->string, "rlBpmChirpCfg_t") != 0)
				{
					goto NEXT;
				}
				cfgCnt = getNumOfConfig(elem1);

				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "chirpStartIdx");
					bpmChpCfg.chirpStartIdx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "chirpEndIdx");
					bpmChpCfg.chirpEndIdx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "constBpmVal");
					bpmChpCfg.constBpmVal = getValueFromJsonElem(elem3);
					
					memcpy(bpmChpCfgPtr, &bpmChpCfg, sizeof(rlBpmChirpCfg_t));
					memset(&bpmChpCfg, 0, sizeof(rlBpmChirpCfg_t));
					bpmChpCfgPtr++;
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlRfDevCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.rfDevCfg, 0, sizeof(gMmwSensCfg.rfCfg.rfDevCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "aeDirection");
				gMmwSensCfg.rfCfg.rfDevCfg.aeDirection = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "aeControl");
				gMmwSensCfg.rfCfg.rfDevCfg.aeControl = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "bssDigCtrl");
				gMmwSensCfg.rfCfg.rfDevCfg.bssDigCtrl = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "aeCrcConfig");
				gMmwSensCfg.rfCfg.rfDevCfg.aeCrcConfig = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfMiscConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.rfMiscCfg, 0, sizeof(gMmwSensCfg.rfCfg.rfMiscCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "miscCtl");
				gMmwSensCfg.rfCfg.rfMiscCfg.miscCtl = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfTxFreqPwrLimitMonConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg, 0, sizeof(gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitLowTx0");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.freqLimitLowTx0 = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitLowTx1");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.freqLimitLowTx1 = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitLowTx2");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.freqLimitLowTx2 = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitHighTx0");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.freqLimitHighTx0 = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitHighTx1");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.freqLimitHighTx1 = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "freqLimitHighTx2");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.freqLimitHighTx2 = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "tx0PwrBackOff");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.tx0PwrBackOff = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "tx1PwrBackOff");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.tx1PwrBackOff = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "tx2PwrBackOff");
				gMmwSensCfg.rfCfg.TxFreqPwrLimitCfg.tx2PwrBackOff = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlDynPwrSave_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.dynPwrSaveCfg, 0, sizeof(gMmwSensCfg.rfCfg.dynPwrSaveCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "blkCfg");
				gMmwSensCfg.rfCfg.dynPwrSaveCfg.blkCfg = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfPhaseShiftCfgs") == 0)
			{
				rlRfPhaseShiftCfg_t phShftCfg = { 0 };
				rlRfPhaseShiftCfg_t *phShftCfgPtr = NULL;

				cfgCnt = getNumOfConfig(temp_elem);
				if (cfgCnt == 0)
				{
					gMmwSensCfg.rfCfg.phaseShiftCfgCnt = cfgCnt;
					goto NEXT;
				}
				else
				{
					gMmwSensCfg.rfCfg.phaseShiftCfgCnt = cfgCnt;
					gMmwSensCfg.rfCfg.phaseShiftCfg = (rlRfPhaseShiftCfg_t*)calloc(cfgCnt, sizeof(rlRfPhaseShiftCfg_t));
					phShftCfgPtr = gMmwSensCfg.rfCfg.phaseShiftCfg;
				}
				elem1 = temp_elem->child;
				if (strcmp(elem1->child->string, "rlRfPhaseShiftCfg_t") != 0)
				{
					goto NEXT;
				}
				
				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "chirpStartIdx");
					phShftCfg.chirpStartIdx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "chirpEndIdx");
					phShftCfg.chirpEndIdx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "tx0PhaseShift");
					phShftCfg.tx0PhaseShift = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "tx1PhaseShift");
					phShftCfg.tx1PhaseShift = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "tx2PhaseShift");
					phShftCfg.tx2PhaseShift = getValueFromJsonElem(elem3);

					memcpy(phShftCfgPtr, &phShftCfg, sizeof(rlRfPhaseShiftCfg_t));
					memset(&phShftCfg, 0, sizeof(rlRfPhaseShiftCfg_t));
					phShftCfgPtr++;
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlInterChirpBlkCtrlCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg, 0, sizeof(gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "rx02RfTurnOffTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx02RfTurnOffTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx13RfTurnOffTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx13RfTurnOffTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx02BbTurnOffTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx02BbTurnOffTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx12BbTurnOffTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx12BbTurnOffTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx02RfPreEnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx02RfPreEnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx13RfPreEnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx13RfPreEnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx02BbPreEnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx02BbPreEnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx13BbPreEnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx13BbPreEnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx02RfTurnOnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx02RfTurnOnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx13RfTurnOnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx13RfTurnOnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx02BbTurnOnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx02BbTurnOnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rx13BbTurnOnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rx13BbTurnOnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rxLoChainTurnOffTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rxLoChainTurnOffTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "txLoChainTurnOffTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.txLoChainTurnOffTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "rxLoChainTurnOnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.rxLoChainTurnOnTime = getValueFromJsonElem(elem1) * 100;
				elem1 = cJSON_GetObjectItem(temp_elem, "txLoChainTurnOnTime_us");
				gMmwSensCfg.rfCfg.interChirpBlkCtrlCfg.txLoChainTurnOnTime = getValueFromJsonElem(elem1) * 100;
			}
			else if (strcmp(temp_elem->string, "rlRfProgFiltConfs") == 0)
			{
				/* TODO : complete this Array, may need to change the struct as for profileCfg for array data */
				rlRfProgFiltConf_t progFiltCfg = { 0 };
				rlRfProgFiltConf_t *progFiltCfgPtr = NULL;

				cfgCnt = getNumOfConfig(temp_elem);
				if (cfgCnt == 0)
				{
					//gMmwSensCfg.rfCfg.progFiltCfg = cfgCnt;
					goto NEXT;
				}
				else
				{
					
				}
				elem1 = temp_elem->child;
				if (strcmp(elem1->child->string, "rlRfProgFiltConf_t") != 0)
				{
					goto NEXT;
				}

				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "profileId");
					gMmwSensCfg.rfCfg.progFiltCfg.profileId = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "coeffStartIdx");
					gMmwSensCfg.rfCfg.progFiltCfg.coeffStartIdx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "progFiltLen");
					gMmwSensCfg.rfCfg.progFiltCfg.progFiltLen = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "progFiltFreqShift_Fs");
					gMmwSensCfg.rfCfg.progFiltCfg.progFiltFreqShift = (getValueFromJsonElem(elem3) / \
						(0.01 * gMmwSensCfg.rfCfg.profileCfg[gMmwSensCfg.rfCfg.progFiltCfg.profileId].digOutSampleRate));

					//memcpy(progFiltCfgPtr, &progFiltCfg, sizeof(rlRfProgFiltConf_t));
					//memset(&phShftCfg, 0, sizeof(rlRfProgFiltConf_t));
					//phShftCfgPtr++;
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rlInterRxGainPhConf_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.interRxGainPhCfg, 0, sizeof(gMmwSensCfg.rfCfg.interRxGainPhCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "profileIndx");
				gMmwSensCfg.rfCfg.interRxGainPhCfg.profileIndx = getValueFromJsonElem(elem1);
				//2nd Gen : "digCompEn"
				//TODO  Complete this
			}
			else if (strcmp(temp_elem->string, "rlTestSource_t") == 0)
			{
				rlTestSourceObject_t *testSrcObj = &gMmwSensCfg.rfCfg.testSrcCfg.testObj[0];
				rlTestSourceAntPos_t *testSrcAntPos = &gMmwSensCfg.rfCfg.testSrcCfg.rxAntPos[0];

				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.testSrcCfg, 0, sizeof(gMmwSensCfg.rfCfg.testSrcCfg));
				
				/*rlTestSourceObjects*/
				cfgCnt = getNumOfConfig(temp_elem->child);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem1 = temp_elem->child->child;
				if (strcmp(elem1->child->string, "rlTestSourceObject_t") != 0)
				{
					goto NEXT;
				}

				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "posX_m");
					testSrcObj->posX = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posY_m");
					testSrcObj->posY = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posZ_m");
					testSrcObj->posZ = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "velX_m_sec");
					testSrcObj->velX = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "velY_m_sec");
					testSrcObj->velY = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "velZ_m_sec");
					testSrcObj->velZ = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "sigLvl_dBFS");
					testSrcObj->sigLvl = (rlUInt16_t)(getValueFromJsonElem(elem3) / -0.1);
					elem3 = cJSON_GetObjectItem(elem2, "posXMin_m");
					testSrcObj->posXMin = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posYMin_m");
					testSrcObj->posYMin = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posZMin_m");
					testSrcObj->posZMin = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posXMax_m");
					testSrcObj->posXMax = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posYMax_m");
					testSrcObj->posYMax = M_CM_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "posZMax_m");
					testSrcObj->posZMax = M_CM_CONVERSION(getValueFromJsonElem(elem3));

					testSrcObj++;
					cfgCnt--;

					elem1 = elem1->next;
				}

				/*rlTestSourceRxAntPos*/
				cfgCnt = getNumOfConfig(temp_elem->child->next);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem1 = temp_elem->child->next->child;
				if (strcmp(elem1->child->string, "rlTestSourceAntPos_t") != 0)
				{
					goto NEXT;
				}

				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "antPosX");
					testSrcAntPos->antPosX = (rlInt8_t)(getValueFromJsonElem(elem3)*8);
					elem3 = cJSON_GetObjectItem(elem2, "antPosZ");
					testSrcAntPos->antPosZ = (rlInt8_t)(getValueFromJsonElem(elem3)*8);

					testSrcAntPos++;
					cfgCnt--;

					elem1 = elem1->next;
				}

				/*2nd Gen : rlTestSourceTxAntPos*/
#if 0
				cfgCnt = getNumOfConfig(temp_elem->child->next->next);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem1 = temp_elem->child->next->next->child;
				if (strcmp(elem1->child->string, "rlTestSourceAntPos_t") != 0)
				{
					goto NEXT;
				}

				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "antPosX");
					testSrcAntPos->antPosX = (rlInt8_t)(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "antPosZ");
					testSrcAntPos->antPosZ = (rlInt8_t)(getValueFromJsonElem(elem3));

					testSrcAntPos++;
					cfgCnt--;

					elem1 = elem1->next;
				}
				//2nd Gen: miscFunCtrl
#endif		
			}
			else if (strcmp(temp_elem->string, "rlRfLdoBypassCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.ldoByPass, 0, sizeof(gMmwSensCfg.rfCfg.ldoByPass));
				elem1 = cJSON_GetObjectItem(temp_elem, "ldoBypassEnable");
				gMmwSensCfg.rfCfg.ldoByPass.ldoBypassEnable = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "supplyMonIrDrop");
				gMmwSensCfg.rfCfg.ldoByPass.supplyMonIrDrop = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "ioSupplyIndicator");
				gMmwSensCfg.rfCfg.ldoByPass.ioSupplyIndicator = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfPALoopbackCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.paLoopBkCfg, 0, sizeof(gMmwSensCfg.rfCfg.paLoopBkCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "paLoopbackFreq_MHz");
				gMmwSensCfg.rfCfg.paLoopBkCfg.paLoopbackFreq = (rlUInt16_t)(100/getValueFromJsonElem(elem1));
				elem1 = cJSON_GetObjectItem(temp_elem, "paLoopbackEn");
				gMmwSensCfg.rfCfg.paLoopBkCfg.paLoopbackEn = (rlUInt8_t)getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfPSLoopbackCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.psLoopBkCfg, 0, sizeof(gMmwSensCfg.rfCfg.psLoopBkCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "psLoopbackFreq_KHz");
				gMmwSensCfg.rfCfg.psLoopBkCfg.psLoopbackFreq = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "psLoopbackEn");
				gMmwSensCfg.rfCfg.psLoopBkCfg.psLoopbackEn = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "psLoopbackTxId");
				gMmwSensCfg.rfCfg.psLoopBkCfg.psLoopbackTxId = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "pgaGainIndex");
				gMmwSensCfg.rfCfg.psLoopBkCfg.pgaGainIndex = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlRfIFLoopbackCfg_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.ifLoopBkCfg, 0, sizeof(gMmwSensCfg.rfCfg.ifLoopBkCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "ifLoopbackFreq");
				gMmwSensCfg.rfCfg.ifLoopBkCfg.ifLoopbackFreq = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "ifLoopbackEn");
				gMmwSensCfg.rfCfg.ifLoopBkCfg.ifLoopbackEn = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlLoopbackBursts") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.loopBackBurstCfg, 0, sizeof(gMmwSensCfg.rfCfg.loopBackBurstCfg));

				/*rlLoopbackBursts*/
				cfgCnt = getNumOfConfig(temp_elem);
				if (cfgCnt == 0)
				{
					goto NEXT;
				}
				elem1 = temp_elem->child;
				if (strcmp(elem1->child->string, "rlLoopbackBurst_t") != 0)
				{
					goto NEXT;
				}

				while (cfgCnt)
				{
					elem2 = elem1->child;
					elem3 = cJSON_GetObjectItem(elem2, "loopbackSel");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.loopbackSel = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "baseProfileIndx");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.baseProfileIndx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "burstIndx");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.burstIndx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "freqConst_GHz");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.freqConst = START_FREQ_HZ_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "slopeConst_MHz_us");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.slopeConst = FREQ_SLOP_KHz_CONVERSION(getValueFromJsonElem(elem3));
					elem3 = cJSON_GetObjectItem(elem2, "txBackoff");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.txBackoff = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "rxGain_dB");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.rxGain = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "txEn");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.txEn = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "bpmConfig");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.bpmConfig = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "digCorrDis");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.digCorrDis = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "ifLbFreq");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.ifLbFreq = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "ifLbMag_10mv");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.ifLbMag = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "ps1PgaIndx");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.ps1PgaIndx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "ps2PgaIndx");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.ps2PgaIndx = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "psLbFreq_KHz");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.psLbFreq = getValueFromJsonElem(elem3);
					elem3 = cJSON_GetObjectItem(elem2, "paLbFreq_MHz");
					gMmwSensCfg.rfCfg.loopBackBurstCfg.paLbFreq = (rlUInt16_t)(100/getValueFromJsonElem(elem3));
					cfgCnt--;

					elem1 = elem1->next;
				}
			}
			else if (strcmp(temp_elem->string, "rllatentFault_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.latentFaultCfg, 0, sizeof(gMmwSensCfg.rfCfg.latentFaultCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "testEn1");
				gMmwSensCfg.rfCfg.latentFaultCfg.testEn1 = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testEn2");
				gMmwSensCfg.rfCfg.latentFaultCfg.testEn2 = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "repMode");
				gMmwSensCfg.rfCfg.latentFaultCfg.repMode = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testMode");
				gMmwSensCfg.rfCfg.latentFaultCfg.testMode = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rlperiodicTest_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.periodicTestCfg, 0, sizeof(gMmwSensCfg.rfCfg.periodicTestCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "periodicity_msec");
				gMmwSensCfg.rfCfg.periodicTestCfg.periodicity = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testEn");
				gMmwSensCfg.rfCfg.periodicTestCfg.testEn = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "repMode");
				gMmwSensCfg.rfCfg.periodicTestCfg.repMode = getValueFromJsonElem(elem1);
			}
			else if (strcmp(temp_elem->string, "rltestPattern_t") == 0)
			{
				/* reset the config structure */
				memset(&gMmwSensCfg.rfCfg.testPatternCfg, 0, sizeof(gMmwSensCfg.rfCfg.testPatternCfg));
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatGenCtrl");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatGenCtrl = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatGenTime");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatGenTime = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatrnPktSize");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatrnPktSize = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatrnPktSize");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatrnPktSize = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "numTestPtrnPkts");
				gMmwSensCfg.rfCfg.testPatternCfg.numTestPtrnPkts = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx0Icfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx0Icfg = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx1Icfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx1Icfg = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx1Qcfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx1Qcfg = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx2Icfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx2Icfg = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx2Qcfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx2Qcfg = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx3Icfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx3Icfg = getValueFromJsonElem(elem1);
				elem1 = cJSON_GetObjectItem(temp_elem, "testPatRx3Qcfg");
				gMmwSensCfg.rfCfg.testPatternCfg.testPatRx3Qcfg = getValueFromJsonElem(elem1);
			}
			else
			{
				//Skip other tags
			}

			NEXT:

			if (temp_elem->child == NULL)
			{
				if (temp_elem->next == NULL)
				{
					//Terminate
					break;
				}
				else
				{
					temp_elem = temp_elem->next;
				}
			}
			else
			{
				temp_elem = temp_elem->next;
			}
		}
	}
	return 0;
}

/** @fn int writeCliCfg()
*
*   @brief This function converts JSON config parameter to CFG file format.
*
*   @param[in] None
*
*   @return success
*/
int writeCliCfg()
{
	int status = 0, idx, retVal = 0, switch_on;
	char cliStr[100] = { 0 };
	char teminatStr = '\n';
	char defaultCliCmd[28][100] = {
					"dfeDataOutputMode ",
					"channelCfg ",
					"adcCfg ",
					"adcbufCfg ",
					"profileCfg ",
					"chirpCfg ",
					"frameCfg ",
					"advFrameCfg ",
					"subFrameCfg ",
					"lowPower ",
					"lvdsStreamCfg ",
					"CQRxSatMonitor ",
					"CQSigImgMonitor ",
					"calibMonCfg ",
					"monCalibReportCfg ",
					"txPowerMonCfg ",
					"txBallbreakMonCfg ",
					"rxGainPhaseMonCfg ",
					"tempMonCfg ",
					"synthFreqMonCfg ",
					"pllConVoltMonCfg ",
					"dualClkCompMonCfg ",
					"rxIfStageMonCfg ",
					"extAnaSigMonCfg ",
					"pmClkSigMonCfg ",
					"rxIntAnaSigMonCfg ",
					"gpadcSigMonCfg ",
					"sensorStart \n\r" };
	char *cfgBanner = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
					  "%  This is auto generated CFG file from JSON file. %\n"
					  "%  Compatible with Studio CLI application ONLY.    %\n"
					  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n";
	extern char *CONVERTED_CFG_FILE_NAME;
	FILE *newCfgFilePtr = fopen(CONVERTED_CFG_FILE_NAME, "w");

	rlProfileCfg_t *profCfgPtr;
	rlChirpCfg_t   *chirpCfgPtr;

	fwrite(cfgBanner, 1, strlen(cfgBanner), newCfgFilePtr);
	/* Write all the initial or default CLI commands to thie file */
	/* dfeDataOutputMode <1> */
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s %d %c", defaultCliCmd[DFE_OUT_MODE], gMmwSensCfg.rfCfg.dfeOutputMode, teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);

	/* channelCfg 15 7 0 */
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s %d %d %d %c", defaultCliCmd[CHANNEL_CFG], gMmwSensCfg.rfCfg.channelCfg.rxChannelEn, \
		gMmwSensCfg.rfCfg.channelCfg.txChannelEn, gMmwSensCfg.rfCfg.channelCfg.cascading,\
		teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);

	/* adcCfg 2 1 */
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s %d %d %c", defaultCliCmd[ADC_CFG], gMmwSensCfg.rfCfg.adcOutCfg.fmt.b2AdcBits, \
		gMmwSensCfg.rfCfg.adcOutCfg.fmt.b2AdcOutFmt, teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);

	/* adcbufCfg -1 0 1 1 1*/
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s %d %d %d %d %d %c", defaultCliCmd[ADC_BUF_CFG], -1, !gMmwSensCfg.datapathCfg.devDataFmtCfg.adcFmt, \
		gMmwSensCfg.datapathCfg.devDataFmtCfg.iqSwapSel, gMmwSensCfg.datapathCfg.devDataFmtCfg.chInterleave, \
		1, teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);

	profCfgPtr = gMmwSensCfg.rfCfg.profileCfg;
	for (idx = 0; idx < gMmwSensCfg.rfCfg.profileCfgCnt; idx++)
	{
		if (profCfgPtr == NULL)
			break;
		/* profileCfg 0 77 429 7 57.14 0 0 70 1 256 5209 0 0 30 */
		memset(cliStr, 0, sizeof(cliStr));
		sprintf(cliStr, "%s %d %.2f %d %d %d %d %d %.2f %d %d %d %d %d %d %c", defaultCliCmd[PROFLIE_CFG], profCfgPtr->profileId, \
			(float)START_FREQ_GHZ_CONVERSION(profCfgPtr->startFreqConst), profCfgPtr->idleTimeConst / 100, \
			profCfgPtr->adcStartTimeConst / 100, profCfgPtr->rampEndTime / 100, profCfgPtr->txOutPowerBackoffCode, \
			profCfgPtr->txPhaseShifter, (float)FREQ_SLOP_MHz_CONVERSION(profCfgPtr->freqSlopeConst), profCfgPtr->txStartTime, \
			profCfgPtr->numAdcSamples, profCfgPtr->digOutSampleRate, profCfgPtr->hpfCornerFreq1, \
			profCfgPtr->hpfCornerFreq2, profCfgPtr->rxGain, teminatStr);
		fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
		profCfgPtr++;
	}
	if (idx == 0)
		retVal = -1;
	chirpCfgPtr = gMmwSensCfg.rfCfg.chirpCfg;
	for (idx = 0; idx < gMmwSensCfg.rfCfg.chirpCfgCnt; idx++)
	{
		if (chirpCfgPtr == NULL)
			break;
		memset(cliStr, 0, sizeof(cliStr));
		sprintf(cliStr, "%s %d %d %d %d %d %d %d %d %c", defaultCliCmd[CHIRP_CFG], chirpCfgPtr->chirpStartIdx, \
			chirpCfgPtr->chirpEndIdx, chirpCfgPtr->profileId, chirpCfgPtr->startFreqVar, chirpCfgPtr->freqSlopeVar, \
			chirpCfgPtr->idleTimeVar / 100, chirpCfgPtr->adcStartTimeVar / 100, chirpCfgPtr->txEnable, teminatStr);
		fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
		chirpCfgPtr++;
	}
	if (idx == 0)
		retVal = -1;
	
	if (gMmwSensCfg.rfCfg.dfeOutputMode == MMW_LEFACY_FRAME_MODE)
	{
		rlFrameCfg_t *frameCfg = &gMmwSensCfg.rfCfg.frameCfg;
		memset(cliStr, 0, sizeof(cliStr));
		sprintf(cliStr, "%s %d %d %d %d %.2f %d %.2f %c", defaultCliCmd[LEG_FRAME_CFG], frameCfg->chirpStartIdx, \
			frameCfg->chirpEndIdx, frameCfg->numLoops, frameCfg->numFrames, (float)((frameCfg->framePeriodicity* 5) / 1000000), \
			frameCfg->triggerSelect, (float)((frameCfg->frameTriggerDelay* 5) / 1000000), teminatStr);
		fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
	}
	else if (gMmwSensCfg.rfCfg.dfeOutputMode == MMW_ADV_FRAME_MODE)
	{
		rlAdvFrameSeqCfg_t *advFramSeqPtr = &gMmwSensCfg.rfCfg.advFrameCfg.frameSeq;
		rlSubFrameCfg_t       *subFramePtr = &gMmwSensCfg.rfCfg.advFrameCfg.frameSeq.subFrameCfg[0];
		memset(cliStr, 0, sizeof(cliStr));
		sprintf(cliStr, "%s %d %d %d %d %d %c", defaultCliCmd[ADV_FRAME_CFG], advFramSeqPtr->numOfSubFrames, \
			advFramSeqPtr->forceProfile, advFramSeqPtr->numFrames, advFramSeqPtr->triggerSelect, \
			advFramSeqPtr->frameTrigDelay, teminatStr);
		fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
		for (idx = 0; idx < RL_MAX_SUBFRAMES; idx++)
		{
			if (subFramePtr == NULL)
				break;
			memset(cliStr, 0, sizeof(cliStr));
			sprintf(cliStr, "%s %d %d %d %d %d %.2f %d %d %d %.2f %c", defaultCliCmd[SUB_FRAME_CFG], idx, subFramePtr->forceProfileIdx, \
				subFramePtr->chirpStartIdx, subFramePtr->numOfChirps, subFramePtr->numLoops, (float)((subFramePtr->burstPeriodicity* 5)/ 1000000), \
				subFramePtr->chirpStartIdxOffset, subFramePtr->numOfBurst, subFramePtr->numOfBurstLoops, \
				(float)((subFramePtr->subFramePeriodicity* 5)/ 1000000), teminatStr);
			fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
		}
	}
	else
	{
		return -1;
	}
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s 0 %d %c", defaultCliCmd[LOW_POWER], gMmwSensCfg.rfCfg.lowPowerCfg.lpAdcMode, teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s -1 0 1 0 %c", defaultCliCmd[LVDS_STREAM], teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
	memset(cliStr, 0, sizeof(cliStr));
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s %d %d %c", defaultCliCmd[CALIB_MON_CFG], gMmwSensCfg.rfCfg.calMonTimeUntCfg.calibMonTimeUnit, \
		gMmwSensCfg.rfCfg.runTimeCalCfg.calibPeriodicity, teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);

	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s 1 1 0 %c", defaultCliCmd[MON_CALIB_REPORT_CFG], /*gMmwSensCfg.rfCfg.calMonTimeUntCfg.calibMonTimeUnit, \
		gMmwSensCfg.rfCfg.runTimeCalCfg.calibPeriodicity, */ teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);

	idx = gMmwSensCfg.monitorCfg.anaMonEnCfg.enMask;
	for (switch_on = 0; switch_on <= RX_MIXER_INPUT_POWER_MONITOR; switch_on++)
	{
		if (!(gMmwSensCfg.monitorCfg.anaMonEnCfg.enMask & (1 << switch_on)))
		{
			continue;
		}
		memset(cliStr, 0, sizeof(cliStr));
		switch (switch_on)
		{
		case TEMPERATURE_MONITOR_EN:
			sprintf(cliStr, "%s 1 %d %c", defaultCliCmd[TEMP_MON_CFG], gMmwSensCfg.monitorCfg.tempMonCfg.tempDiffThresh, teminatStr);
			break;
		case RX_GAIN_PHASE_MONITOR_EN:
			sprintf(cliStr, "%s 1 %d %c", defaultCliCmd[RX_GAIN_PH_MON_CFG], \
				gMmwSensCfg.monitorCfg.rxGainPhMonCfg.profileIndx, teminatStr);
			break;
		case RX_NOISE_MONITOR_EN:
			break;
		case RX_IFSTAGE_MONITOR_EN:
			sprintf(cliStr, "%s 1 %d %c", defaultCliCmd[RX_IF_STAGE_MON_CFG], gMmwSensCfg.monitorCfg.rxIfStageMonCfg.profileIndx, teminatStr);
			break;
		case TX0_POWER_MONITOR_EN:
			sprintf(cliStr, "%s 1 0 %d %c", defaultCliCmd[TX_POWER_MON_CFG], gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[0].profileIndx, teminatStr);
			break;
		case TX1_POWER_MONITOR_EN:
			sprintf(cliStr, "%s 1 1 %d %c", defaultCliCmd[TX_POWER_MON_CFG], gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[1].profileIndx, teminatStr);
			break;
		case TX2_POWER_MONITOR_EN:
			sprintf(cliStr, "%s 1 2 %d %c", defaultCliCmd[TX_POWER_MON_CFG], gMmwSensCfg.monitorCfg.txIntAnaSigMonCfg[2].profileIndx, teminatStr);
			break;
		case TX0_BALLBREAK_MONITOR_EN:
			sprintf(cliStr, "%s 1 0 %c", defaultCliCmd[TX_BALL_MON_CFG], teminatStr);
			break;
		case TX1_BALLBREAK_MONITOR_EN:
			sprintf(cliStr, "%s 1 1  %c", defaultCliCmd[TX_BALL_MON_CFG], teminatStr);
			break;
		case TX2_BALLBREAK_MONITOR_EN:
			sprintf(cliStr, "%s 1 2 %c", defaultCliCmd[TX_BALL_MON_CFG], teminatStr);
			break;
		case TX_GAIN_PHASE_MONITOR_EN:
			break;
		case TX0_BPM_MONITOR_EN:
			break;
		case TX1_BPM_MONITOR_EN:
			break;
		case TX2_BPM_MONITOR_EN:
			break;
		case SYNTH_FREQ_MONITOR_EN:
			sprintf(cliStr, "%s 1 0 %c", defaultCliCmd[SYNTH_FREQ_MON_CFG], teminatStr);
			break;
		case EXTERNAL_ANALOG_SIGNALS_MONITOR_EN:
			sprintf(cliStr, "%s 1 %c", defaultCliCmd[EXT_ANA_SIG_MON_CFG], teminatStr);
			break;
		case INTERNAL_TX0_SIGNALS_MONITOR_EN:
			break;
		case INTERNAL_TX1_SIGNALS_MONITOR_EN:
			break;
		case INTERNAL_TX2_SIGNALS_MONITOR_EN:
			break;
		case INTERNAL_RX_SIGNALS_MONITOR_EN:
			sprintf(cliStr, "%s 1 %d %c", defaultCliCmd[RX_INT_ANA_MON_CFG], gMmwSensCfg.monitorCfg.rxIntAnaSigMonCfg.profileIndx, teminatStr);
			break;
		case INTERNAL_PMCLKLO_SIGNALS_MONITOR_EN:
			sprintf(cliStr, "%s 1 %d %c", defaultCliCmd[PM_CLK_SIG_MON_CFG], gMmwSensCfg.monitorCfg.pmClkAnaSigMonCfg.profileIndx, teminatStr);
			break;
		case INTERNAL_GPADC_SIGNALS_MONITOR_EN:
			sprintf(cliStr, "%s 1 %c", defaultCliCmd[GPADC_SIG_MON_CFG], teminatStr);
			break;
		case PLL_CONTROL_VOLTAGE_MONITOR_EN:
			sprintf(cliStr, "%s 1 %c", defaultCliCmd[PLL_CTRL_MON_CFG], teminatStr);
			break;
		case DCC_CLOCK_FREQ_MONITOR_EN:
			sprintf(cliStr, "%s 1 %c", defaultCliCmd[DUAL_CLK_MON_CFG], teminatStr);
			break;
		case RX_IF_SATURATION_MONITOR_EN:
			sprintf(cliStr, "%s 1 %d %d %d %d %d %c", defaultCliCmd[CQ_SAT_MON_CFG], gMmwSensCfg.monitorCfg.rxSatMonCfg.profileIndx, \
				gMmwSensCfg.monitorCfg.rxSatMonCfg.satMonSel, gMmwSensCfg.monitorCfg.rxSatMonCfg.primarySliceDuration, \
				gMmwSensCfg.monitorCfg.rxSatMonCfg.numSlices, gMmwSensCfg.monitorCfg.rxSatMonCfg.rxChannelMask, teminatStr);
			break;
		case RX_SIG_IMG_BAND_MONITORING_EN:
			sprintf(cliStr, "%s %d %d %d %c", defaultCliCmd[CQ_IMG_MON_CFG], gMmwSensCfg.monitorCfg.sigImgMonCfg.profileIndx, \
				gMmwSensCfg.monitorCfg.sigImgMonCfg.numSlices, gMmwSensCfg.monitorCfg.sigImgMonCfg.timeSliceNumSamples, teminatStr);
			break;
		case RX_MIXER_INPUT_POWER_MONITOR:
			break;
		default:
			break;
		}
		fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
	}
	memset(cliStr, 0, sizeof(cliStr));
	sprintf(cliStr, "%s %c", defaultCliCmd[SENSOR_START], teminatStr);
	fwrite(cliStr, 1, strlen(cliStr), newCfgFilePtr);
	/* close the cfg file */
	fclose(newCfgFilePtr);
}

/** @fn int mmw_jsonToCfgConversion()
*
*   @brief This function converts JSON to CFG format.
*			It needs to parse whole JSON file and generate a cfg file 
*			having CLI commands in it
*
*   @param[in] None
*
*   @return success
*/
int mmw_jsonToCfgConversion()
{
	cJSON *json_name = NULL;
	cJSON *resolution = NULL;
	cJSON *resolutions = NULL;
	int status = 0, idx, retVal = 0, switch_on;
	resolutions = cJSON_GetObjectItem(mmwave_jsonParseOp, "mmWaveDevices");
	cJSON_ArrayForEach(resolution, resolutions)
	{
		status = mmw_rfConfigFromJson(cJSON_GetObjectItem(resolution, "rfConfig"));
		if (status != 0)
		{
		}
		status = mmw_dataCaptureCfgFromJson(cJSON_GetObjectItem(resolution, "rawDataCaptureConfig"));
		if (status != 0)
		{
		}
		status = mmw_monitorCfgFromJson(cJSON_GetObjectItem(resolution, "monitoringConfig"));
		if (status != 0)
		{
		}
	}
	writeCliCfg();
	return 0;
}

int mmw_jsonParser(char *filePath)
{
	mmwave_jsonParseOp = parse_file(filePath);
	mmw_jsonToCfgConversion();
	return 0;
}
