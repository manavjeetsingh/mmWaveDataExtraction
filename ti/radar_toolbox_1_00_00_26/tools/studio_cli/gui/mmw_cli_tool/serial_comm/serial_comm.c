/****************************************************************************************
* FileName     : serial_comm.c
*
* Description  : This file implements Serial COM Port Read/Write functions
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

#include <stdio.h>
#include "mmw_config.h"
#include "serial_comm.h"

#define BUFFERLENGTH				 256
#define	MAX_READ_COM_RETRY			 100
#define READ_SIZE_BEFORE_WRITE       20
#define DELAY_TO_READ_UART_RESP_MSEC 100

/** @fn int findWordInSentence(char *word, char *sentence)
*
*   @brief Search the given Word in the Sentence 
*
*   @param[in] input buffer
*   @param[in] input senstence to search

*   @return -1: if not found, else starting location of word in sentence. 
*/
int findWordInSentence(char *word, char *sentence)
{
	int n = 0;
	int m = 0;
	int foundLoc = 0;
	int len = strlen(word); 
	int retVal = -1;

	while (sentence[n] != '\0') {
		/* if first character of search string matches */
		if (sentence[n] == word[m]) {    
			/* keep on searching till end */
			while (sentence[n] == word[m] && sentence[n] != '\0') {
				n++;
				m++;
			}

			/* if we sequence of characters matching with the length of searched string */
			if (m == len)
			{
				/* we find our search string */
				foundLoc++;
				retVal = n - strlen(word) +1;
				break;
			}
		}

		n++;
		/* reset the counter to start from first character of the search string. */
		m = 0;  
	}

	return retVal;
}


/** @fn int serialComm_Close(HANDLE hSerial)
*
*   @brief Close Serial COM Port
*
*   @param[in] serial com handle

*   @return success
*/
int serialComm_Close(HANDLE hSerial)
{
	if (hSerial != NULL)
	{
		/* Closing the Serial Port */
		CloseHandle(hSerial);
	}
	return 0;
}


/** Debug funtion */
void HandleASuccessfulRead(char *inputStr, int length)
{
	int i=0;

	while(length)
	{
		PRINT_FUNC("%x", inputStr[i]);
		i++;
		length--;
	}
}

/** @fn int serialComm_Write(HANDLE hComm, char *cmdStr)
*
*   @brief Write the given data over Serial COM port
*
*   @param[in] serial com handle
*   @param[in] input buffer
*
*   @return success or error
*/
int serialComm_Write(HANDLE hComm, char *cmdStr)
{
	DWORD  dNoOfBytesWritten = 0, NoBytesRead = 0;
	// No of bytes to write into the port
	DWORD  dNoOFBytestoWrite = strlen(cmdStr);
	char rdBuff[50] = { 0 };
	int doneRspLoc, errorRsp, retVal, rdCharCnt, rdCnt;

	/* write the given buffer over serial port */
	if (!WriteFile(hComm, cmdStr, dNoOFBytestoWrite, &dNoOfBytesWritten, NULL))
	{
		PRINT_FUNC("Error writing text to COMM\n");
		if (GetLastError() == ERROR_IO_PENDING)
		{
			return dNoOFBytestoWrite;
		}
	}
	else
	{
		PRINT_FUNC("\n[WR]%s [%d]\n", cmdStr, dNoOfBytesWritten);
	}
	/* delay a bit before reading response of CLI CMD */
	Sleep(500);
	/* reading few bytes before writing to UART to clear the buffer */
	if (!ReadFile(hComm, &rdBuff[0], dNoOFBytestoWrite+READ_SIZE_BEFORE_WRITE, &NoBytesRead, NULL))
	{
		PRINT_FUNC("wrong character");
	}

	rdCnt = NoBytesRead;
	/* Search "Done" or "Error" in the received Text */
	doneRspLoc = findWordInSentence("Done", &rdBuff[0]);
	errorRsp = findWordInSentence("Error", &rdBuff[0]);

	if ((doneRspLoc > 0) || (errorRsp > 0))
	{
		char errCodeStr[20] = { 0 };
		int  strCnt = 0;
		rdCharCnt = (doneRspLoc > 0) ? (doneRspLoc + strlen("Done")) : \
			(errorRsp + strlen("Error"));
		/* read till '\n' after "Done" or "Error" location from COM Port */
		while (findWordInSentence("\n", &rdBuff[rdCharCnt - 1]) < 0)
		{
			/* this string contain error code if it has "Error" */
			ReadFile(hComm, &rdBuff[rdCnt], 4, &NoBytesRead, NULL);

			rdCnt += NoBytesRead;
		}
		/* if error string recieved */
		if (errorRsp > 0)
		{
			strncpy(&errCodeStr[0], &rdBuff[rdCharCnt], (rdCnt - rdCharCnt));
			retVal = atoi(errCodeStr);
			/* If this is error then print the error value */
			printf("[ERROR] mmwave device returns [%d]\n", atoi(errCodeStr));
			return retVal;
		}
		else
		{
			/* command executed sucessfully */
			printf("Done\n\r");
		}
	}

	return dNoOfBytesWritten;
}

/** @fn int serialComm_Read(HANDLE hComm, char *rspStr, int length)
*
*   @brief Write the given data over Serial COM port
*
*   @param[in] serial com handle
*   @param[out] input buffer
*   @param[in] input buffer length
*
*   @return success or error
*/
int serialComm_Read(HANDLE hComm, char *rspStr, int length)
{
	DWORD NoBytesRead=0;                     
	/* read the data from serial port */
	if (!ReadFile(hComm, rspStr, length, &NoBytesRead, NULL))
	{
		PRINT_FUNC("wrong character");
		return -1;
	}

	return NoBytesRead;
}

/** @fn HANDLE serialComm_Setup(char *pcCommPort)
*
*   @brief Setup the serial comm port
*
*   @param[in] serial com port number string
*
*   @return COMM handle
*/
HANDLE serialComm_Setup(char *pcCommPort)
{
	HANDLE hComm;
	COMSTAT ComStat;
	DWORD dwBytesRead = 1024;
	DWORD dwErrorFlags;
	DCB dcbSerialParams;
	COMMTIMEOUTS timeouts = { 0 };
	BOOL Write_Status;

	hComm = CreateFileA(pcCommPort,
		GENERIC_READ | GENERIC_WRITE,
		0,    // must be opened with exclusive-access
		NULL, // no security attributes
		OPEN_EXISTING, // must use OPEN_EXISTING
		NULL, //FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OVERLAPPED
		NULL  // hTemplate must be NULL for comm devices
	);

	if (hComm == INVALID_HANDLE_VALUE)
	{
		if (GetLastError() == ERROR_FILE_NOT_FOUND)
		{
			puts("[ERROR] cannot open serial port!");
			return NULL;
		}

		puts("[ERROR] cannot open serial port!");
		return NULL;
	}
	else
	{
		PRINT_FUNC("opening serial port successful");
	}

	/* The size of input buffer and output buffer is 1024 */
	SetupComm(hComm, 1024, 1024);

	/* Set COM port timeout settings */
	timeouts.ReadIntervalTimeout = 1;
	timeouts.ReadTotalTimeoutConstant = 1;
	timeouts.ReadTotalTimeoutMultiplier = 1;
	timeouts.WriteTotalTimeoutConstant = 1;
	timeouts.WriteTotalTimeoutMultiplier = 1;
	SetCommTimeouts(hComm, &timeouts);

	dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
	/* retreives  the current settings */
	Write_Status = GetCommState(hComm, &dcbSerialParams);     
	if (Write_Status == FALSE) {
		PRINT_FUNC("[ERROR] in GetCommState()");
		CloseHandle(hComm);
		return NULL;
	}

	// Setting BaudRate = 921600
	dcbSerialParams.BaudRate = MMW_BAUD_RATE;
	// Setting ByteSize = 8
	dcbSerialParams.ByteSize = 8;
	// Setting StopBits = 1
	dcbSerialParams.StopBits = ONESTOPBIT;
	// Setting Parity = None
	dcbSerialParams.Parity = NOPARITY;
	//Configuring the port according to settings in DCB
	Write_Status = SetCommState(hComm, &dcbSerialParams);

	if (Write_Status == FALSE)
	{
		PRINT_FUNC("\n   Error! in Setting DCB Structure");
		CloseHandle(hComm);
		return NULL;
	}

	PurgeComm(hComm, PURGE_TXCLEAR | PURGE_RXCLEAR);
	/* clear any existing error */
	ClearCommError(hComm, &dwErrorFlags, &ComStat);

	if (!EscapeCommFunction(hComm, CLRDTR))
		printf("clearing DTR");
	Sleep(200);
	if (!EscapeCommFunction(hComm, SETDTR))
		printf("setting DTR");

	return hComm;
}


/** @fn int serialComm_CmdWrRd(HANDLE hComm, char ccsDebugEn)
*
*   @brief Write CLI Command from cfg file and readback response.
*
*   @param[in] serial com port handle
*   @param[in] Optional parameter when app is debugged in CCS
*
*   @return success or error code
*/
int serialComm_CmdWrRd(HANDLE hComm, char ccsDebugEn)
{
	// Buffer Containing Rxed Data
	char  SerialBuffer[BUFFERLENGTH + 1] = { 0 };
	DWORD NoBytesRead;
	int i = 0, j = 0, found_result = MAX_READ_COM_RETRY, 
		rdStrLen = 0, retVal = 0;
	// No of bytes to write into the port
	DWORD  dNoOFBytestoWrite = 0;              
	// No of bytes written to the port
	DWORD  dNoOfBytesWritten = 0;          
	char cliCmdStr[BUFFERLENGTH + 1] = { 0 };
	COMSTAT ComStat;
	DWORD dwBytesRead = 1024, dwErrorFlags;
	/* Write COM POrt data in loop for all the CLI command in cfg file */
	do
	{
		/* clear the command string buffer */
		memset(&cliCmdStr[0], 0, sizeof(cliCmdStr));
		/* read CLI CMD, one line at a time from cfg file */
		retVal = readCliCmdFromCfgFile(&cliCmdStr[0]);
		if (retVal <= 0)
			break;

		dNoOFBytestoWrite = strlen(cliCmdStr);

		/* reading few bytes before writing to UART to clear the buffer */
		if (!ReadFile(hComm, &SerialBuffer[0], READ_SIZE_BEFORE_WRITE, &NoBytesRead, NULL))
		{
			PRINT_FUNC("wrong character");
			break;
		}
		/* write the CLI CMD over serial port */
		if (!WriteFile(hComm, &cliCmdStr[0], dNoOFBytestoWrite, &dNoOfBytesWritten, NULL))
		{
			PRINT_FUNC("Error writing text to Serial Port\n");
			if (GetLastError() == ERROR_IO_PENDING)
			{
				return dNoOFBytestoWrite;
			}
		}
		else
		{
			PRINT_FUNC("\n[WR]%s [%d]\n", cliCmdStr, dNoOfBytesWritten);
		}

		/* reading the COM port contains the lastly written data as well,
		 * we read those length and extra bytes */
		rdStrLen = dNoOFBytestoWrite + 7;

		/* Reset the read buffer */
		memset(&SerialBuffer[0], 0, BUFFERLENGTH + 1);

		Sleep(DELAY_TO_READ_UART_RESP_MSEC*(1+ccsDebugEn));
		int rdCnt = 0;
		int errorRsp, doneRspLoc;
		char rdCharCnt = 0;
		/* loop till we are able to read response from device */
		while (found_result)
		{
			doneRspLoc = -1;
			errorRsp = -1;

			/* if device is not responding in few cycles then wait a little before next read
			 * Usually SensorStart command takes more time as it call all BSS config APIs
			 * for sensorStart CLI command */
			if (found_result < (MAX_READ_COM_RETRY-1))
				Sleep(DELAY_TO_READ_UART_RESP_MSEC*(1 + ccsDebugEn));

			/********* Read COM port data ******************/
			if (!ReadFile(hComm, &SerialBuffer[rdCnt], rdStrLen, &NoBytesRead, NULL))
			{
				PRINT_FUNC("wrong character");
				retVal = -1;
				goto EXIT;
			}
			else if (NoBytesRead > 0)
			{
				rdCnt += NoBytesRead;
				if (rdCnt > sizeof(SerialBuffer))
				{
					retVal = -1;
					goto EXIT;
				}
				/* read 10 bytes at a time in next iteration */
				rdStrLen = 10;

				PRINT_FUNC("[RD]%s", SerialBuffer);
				
				/* Search "Done" or "Error" in the received Text */
				doneRspLoc = findWordInSentence("Done", &SerialBuffer[0]);
				errorRsp = findWordInSentence("Error", &SerialBuffer[0]);
				
				if ((doneRspLoc > 0) || (errorRsp > 0))
				{
					char errCodeStr[10] = { 0 };
					int  strCnt = 0;
					rdCharCnt = (doneRspLoc > 0) ? (doneRspLoc +strlen("Done")): \
								(errorRsp + strlen("Error"));
					/* read till '\n' after "Done" or "Error" location from COM Port */
					while (findWordInSentence("\n", &SerialBuffer[rdCharCnt-1]) < 0)
					{
						/* this string contain error code if it has "Error" */
						ReadFile(hComm, &SerialBuffer[rdCnt], 4,\
							&NoBytesRead, NULL);
						
						rdCnt += NoBytesRead;
					}
					/* if error string recieved */
					if (errorRsp > 0)
					{
						strncpy(&errCodeStr[0], &SerialBuffer[rdCharCnt], (rdCnt - rdCharCnt));
						retVal = atoi(errCodeStr);
						/* If this is error then print the error value */
						printf("[ERROR] mmwave device returns [%d]\n", atoi(errCodeStr));
						goto EXIT;
					}
					else
					{
						/* command executed sucessfully */
						printf("Done\n\r");
					}
					break;
				}
			}
			else
			{
				/* do nothing */
			}
			/* decrement the retry count */
			found_result--;

			/* it is in last retry to read Done/Error msg and not yet got RSP */
			if (found_result == 0)
			{
				retVal = -1;
				printf("[ERROR] mmwave device NO RSP for CLI CMD\n");
				goto EXIT;
			}
		}
		/* set retry count to max defined */
		found_result = MAX_READ_COM_RETRY;
	} while (1);
EXIT:
	return retVal;
}
