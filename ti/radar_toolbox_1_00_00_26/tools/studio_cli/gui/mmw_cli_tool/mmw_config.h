/****************************************************************************************
* FileName     : mmw_config.h
*
* Description  : This file implements mmwave link example application.
*
****************************************************************************************
* (C) Copyright 2014, Texas Instruments Incorporated. - TI web address www.ti.com
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
*******************************************************************************
*/
#ifndef MMW_CONFIG_H
#define MMW_CONFIG_H

#include <ti/control/mmwavelink/mmwavelink.h>
#include <windows.h>

#ifdef DEBUG_PRINTF_EN
#define PRINT_FUNC    printf
#else
#define PRINT_FUNC
#endif

/******************************************************************************
* MACROS
*******************************************************************************
*/
/*string length for reading from config file*/
#define STRINGLEN	 256
#define CLI_CMD_LEN  50
#define MAX_ADC_DATA_FILE_NAME		200
#define MAX_ADC_FILE_COLLECTED		1
#define MMWAVE_CONFIG_CFG_FORMAT    1
#define MMWAVE_CONFIG_JSON_FORMAT   0
/* COM Port Baud Rate */
#define MMW_BAUD_RATE				 921600

#define MAX_FILE_PATH_LEN           255
#define MAX_MONITORING_REPORT_INTERATION   10000
/* to indicate the monitor capture is triggered/stopped */
#define MONITOR_CAPTURE_TRIGGERED    1 
#define MONITOR_CAPTURE_STOPPED      2
#define MODULE_NOT_EXECUTING   0
#define MODULE_EXECUTING	   1

#define MMW_DEV_SENSOR_START        1
#define MMW_DEV_SENSOR_STOP         2

#define DCA_NOT_INITIATED           0
#define DCA_CONFIGURED              1
#define DCA_CAPTURE_TRIG            2
#define DCA_CAPTURE_STOP            3


#define TEMPERATURE_MONITOR_EN                   (0)
#define RX_GAIN_PHASE_MONITOR_EN                 (1)
#define RX_NOISE_MONITOR_EN                      (2)
#define RX_IFSTAGE_MONITOR_EN                    (3)
#define TX0_POWER_MONITOR_EN                     (4)
#define TX1_POWER_MONITOR_EN                     (5)
#define TX2_POWER_MONITOR_EN                     (6)
#define TX0_BALLBREAK_MONITOR_EN                 (7)
#define TX1_BALLBREAK_MONITOR_EN                 (8)
#define TX2_BALLBREAK_MONITOR_EN                 (9)
#define TX_GAIN_PHASE_MONITOR_EN                 (10)
#define TX0_BPM_MONITOR_EN                       (11)
#define TX1_BPM_MONITOR_EN                       (12)
#define TX2_BPM_MONITOR_EN                       (13)
#define SYNTH_FREQ_MONITOR_EN                    (14)
#define EXTERNAL_ANALOG_SIGNALS_MONITOR_EN       (15)
#define INTERNAL_TX0_SIGNALS_MONITOR_EN          (16)
#define INTERNAL_TX1_SIGNALS_MONITOR_EN          (17)
#define INTERNAL_TX2_SIGNALS_MONITOR_EN          (18)
#define INTERNAL_RX_SIGNALS_MONITOR_EN           (19)
#define INTERNAL_PMCLKLO_SIGNALS_MONITOR_EN      (20)
#define INTERNAL_GPADC_SIGNALS_MONITOR_EN        (21)
#define PLL_CONTROL_VOLTAGE_MONITOR_EN           (22)
#define DCC_CLOCK_FREQ_MONITOR_EN                (23)
#define RX_IF_SATURATION_MONITOR_EN              (24)
#define RX_SIG_IMG_BAND_MONITORING_EN            (25)
#define RX_MIXER_INPUT_POWER_MONITOR             (26)

/******************************************************************************
* mmwave sensor config STRUCTURES
******************************************************************************
*/

typedef enum cldCmdStrIdx
{
	DFE_OUT_MODE = 0,
	CHANNEL_CFG,
	ADC_CFG,
	ADC_BUF_CFG,
	PROFLIE_CFG,
	CHIRP_CFG,
	LEG_FRAME_CFG,
	ADV_FRAME_CFG,
	SUB_FRAME_CFG,
	LOW_POWER,
	LVDS_STREAM,
	CQ_SAT_MON_CFG,
	CQ_IMG_MON_CFG,
	ANA_MON_CFG,
	CALIB_MON_CFG,
	MON_CALIB_REPORT_CFG,
	TX_POWER_MON_CFG,
	TX_BALL_MON_CFG,
	RX_GAIN_PH_MON_CFG,
	TEMP_MON_CFG,
	SYNTH_FREQ_MON_CFG,
	PLL_CTRL_MON_CFG,
	DUAL_CLK_MON_CFG,
	RX_IF_STAGE_MON_CFG,
	EXT_ANA_SIG_MON_CFG,
	PM_CLK_SIG_MON_CFG,
	RX_INT_ANA_MON_CFG,
	GPADC_SIG_MON_CFG,
	SENSOR_START
}CLI_CMD_STR_IDX_e;

typedef enum
{
	MMW_LEFACY_FRAME_MODE =1,
	MMW_CONTINUOUS_MODE,
	MMW_ADV_FRAME_MODE	
}DFE_OUTPUT_MODE_e;

/* this is to store each module execution status */
typedef struct moduleExecutionState
{
	/* 0: No Monitor Capture, 1: Monitor Capture running */
	uint8_t monCaptureState;
	/* 0: DCA is not initiated, 1 : DCA configured, 2 : DCA capture triggered */
	uint8_t dcaCaptureState;
	uint8_t mmwDevCfgState;
	uint8_t postProcState;
}moduleExecutionState_t;

/* Stored the Monitoring report counts for each type */
typedef struct rcvdReportCount
{
	int initCalCnt;
	int runTimeCnt;
	int digLatentCnt;
	int tempMonCnt;
	int rxGainMonCnt;
	int rxIfStageCnt;
	int tx0PwrCnt;
	int tx1PwrCnt;
	int tx2PwrCnt;
	int tx0BallCnt;
	int tx1BallCnt;
	int tx2BallCnt;
	int txGainCnt;
	int synthFreqCnt;
	int extAnaCnt;
	int pmClkCnt;
	int gpadcCnt;
	int pllCtrlCnt;
	int dccFreqCnt;
	int rxIntAnaCnt;
}rcvdReportCount_t;

typedef struct rfConfig
{
	DFE_OUTPUT_MODE_e			dfeOutputMode;/* advancedFrameChirp/ legacyFrameChirp/ Continuous mode*/
	rlChanCfg_t					channelCfg;
	rlAdcOutCfg_t				adcOutCfg;
	rlLowPowerModeCfg_t			lowPowerCfg;

	rlRfCalMonTimeUntConf_t		calMonTimeUntCfg;

	rlRfCalMonFreqLimitConf_t	rfCalMonFreqLimitCfg;
	//rlRfApllSynthBWCtlConf_t	rfApllSynthCtlCfg;
	rlRfInitCalConf_t			rfInitCalCfg;
	rlRunTimeCalibConf_t		runTimeCalCfg;
	rlFrameCfg_t				frameCfg;
	rlAdvFrameCfg_t				advFrameCfg;
	rlContModeCfg_t				contModeCfg;
	rlContModeEn_t				contModeEn;
	rlRfDevCfg_t				rfDevCfg;
	rlRfMiscConf_t				rfMiscCfg;
	rlRfTxFreqPwrLimitMonConf_t TxFreqPwrLimitCfg;
	rlDynPwrSave_t				dynPwrSaveCfg;

	rlProfileCfg_t				*profileCfg;
	unsigned short				profileCfgCnt;
	rlChirpCfg_t				*chirpCfg;
	unsigned short				chirpCfgCnt;
	rlBpmChirpCfg_t				*bpmChirpCfg;
	unsigned short				bpmCfgCnt;
	rlRfPhaseShiftCfg_t			*phaseShiftCfg;
	unsigned short				phaseShiftCfgCnt;

	rlInterChirpBlkCtrlCfg_t	interChirpBlkCtrlCfg;
	rlRfProgFiltConf_t			progFiltCfg;
	rlInterRxGainPhConf_t		interRxGainPhCfg;

	rlTestSource_t				testSrcCfg;

	rlRfLdoBypassCfg_t			ldoByPass;

	rlRfPALoopbackCfg_t			paLoopBkCfg;
	rlRfPSLoopbackCfg_t			psLoopBkCfg;
	rlRfIFLoopbackCfg_t			ifLoopBkCfg;
	rlLoopbackBurst_t			loopBackBurstCfg;
	rllatentFault_t				latentFaultCfg;
	rlperiodicTest_t			periodicTestCfg;
	rltestPattern_t				testPatternCfg;

}rfConfig_t;

typedef struct dataPathConfig
{
	rlDevDataFmtCfg_t			devDataFmtCfg;
	rlDevDataPathCfg_t			devDataPathCfg;
	rlDevLaneEnable_t			devLaneEnCfg;
	rlDevDataPathClkCfg_t		devDataPathClkCfg;
	rlDevLvdsLaneCfg_t			devLvdsLaneCfg;
	rlDevHsiClk_t				devHsiClkCfg;
}dataPathConfig_t;

typedef struct monitoringCfg
{
	/* monitoring APIs */
	rlMonAnaEnables_t				anaMonEnCfg;
	rlDigMonPeriodicConf_t			digPeriodMonCfg;
	rlTempMonConf_t					tempMonCfg;
	rlRxGainPhaseMonConf_t			rxGainPhMonCfg;
	rlRxNoiseMonConf_t				rxNoiseMonCfg;
	rlRxIfStageMonConf_t			rxIfStageMonCfg;
	rlSynthFreqMonConf_t			synthFreqMonCfg;
	rlExtAnaSignalsMonConf_t		extAnaSigMonCfg;
	rlTxIntAnaSignalsMonConf_t		txIntAnaSigMonCfg[3];
	rlAllTxIntAnaSignalsMonConf_t	allTxIntAnaSigMonCfg;
	rlTxPowMonConf_t				txPowerMonCfg[3];
	rlAllTxPowMonConf_t				allTxPowerMonCfg;
	rlTxBallbreakMonConf_t			txBallBreakMonCfg[3];
	rlAllTxBallBreakMonCfg_t		allTxBallBreakMonCfg;
	rlTxGainPhaseMismatchMonConf_t  txGainPhMisMonCfg;
	rlRxIntAnaSignalsMonConf_t		rxIntAnaSigMonCfg;
	rlPmClkLoIntAnaSignalsMonConf_t pmClkAnaSigMonCfg;
	rlGpadcIntAnaSignalsMonConf_t	gpadcAnaSigMonCfg;
	rlPllContrVoltMonConf_t			pllCtrlVoltMonCfg;
	rlDualClkCompMonConf_t			dualClkComMonCfg;
	rlRxSatMonConf_t				rxSatMonCfg;
	rlSigImgMonConf_t				sigImgMonCfg;
	rlRxMixInPwrMonConf_t			rxMixInPwrMonCfg;

}monitoringCfg_t;


typedef struct mmwave_sensor_config
{
	rfConfig_t		 rfCfg;
	dataPathConfig_t datapathCfg;
	monitoringCfg_t  monitorCfg;
}mmwave_sensor_config_t;

typedef struct mmw_configInfo_t
{
	int calMonTimeUnit;
	int deviceType;
}mmw_configInfo;

typedef enum MMWAVE_DEVICE_VAR
{
	AWR1243 = 0,
	AWR2243,
	AWR1443,
	AWR1642,
	AWR1843,
	AWR6843,
	IWR1443,
	IWR1642,
	IWR1843,
	IWR6843
}MMWAVE_DEVICE_VARIANT;

typedef struct rlDcaConfig
{
	unsigned short packetDelay_us;

	unsigned short DCA1000ConfigPort;

	unsigned short DCA1000DataPort;

	unsigned short maxRecFileSize_MB;

	unsigned char  sequenceNumberEnable;
	unsigned char  captureStopMode; /* 0: infinite */
	unsigned char  lvdsLaneMode; /* 2 or 4 lanes */
	unsigned char  dataFormat; /* 12/14/16 bit mode */
	unsigned int   durationToCapture_ms;

	unsigned int   bytesToCapture;

	unsigned int   framesToCapture;

	unsigned char DCA1000IPAddress[20];
	unsigned char systemIPAddress[20]; //TODO JIT: may read by the app, pc own IP addr 
	unsigned char DCA1000MacAddress[20];
	/**
	 * @brief  Path to store Captured unsigned binary file (CAPTURED_ADC_DATA_PATH)
	 */
	char adcDataPath[MAX_FILE_PATH_LEN];

	char filePrefix[MAX_FILE_PATH_LEN];

}rlDcaConfig_t;

/*! \brief
* Global Configuration Structure
*/
typedef struct rlDevGlobalCfg
{
	/**
	 * @brief  DCA capture enable/disable (Enable_DCA_Capture)
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char dcaCaptureEn;
	/**
	 * @brief  mmWave Configuration enable/disable (Enable_Config_Mmwave)
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char mmWaveConfigEn;
	/**
	 * @brief  Monitoring Report Capture enable/disable (Enable_Monitor_Capture)
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char monitorCaptureEn;
	/**
	 * @brief  No. of Monitoring Report to capture.
	 */
	unsigned int totalReportToStore;
	/**
	 * @brief  Post Process of captured data enable/disable (Enable_PostProc)
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char postProcEn;
	/**
	 * @brief  mmWave Config fiel format (CONFIG_FILE_FORMAT)
	 *         1 - cfg; 0 - json
	 */
	unsigned char mmwCfgFormat;
	/**
	 * @brief  Device COM port number (COM_PORT_NUM)
	 */
	unsigned char comPortNum;
	
	MMWAVE_DEVICE_VARIANT mmwaveDevVariant;

	/**
	 * @brief mmwave Sensor Device Variant
	 */
	char mmwaveDevice[8];
	/**
	 * @brief  mmwave configuration file path (CONFIG_JSON_PATH)
	 */
	char mmwConfigPath[MAX_FILE_PATH_LEN];

	/**
	 * @brief  Path to store the monitoring report in JSON format (MONITORING_JSON_PATH)
	 */
	char monReportPath[MAX_FILE_PATH_LEN];

	/**
	 * @brief Path for PostProc tool (POST_PROC_EXE_PATH)
	 */
	char postProcToolPath[MAX_FILE_PATH_LEN];

	/**
	 * @brief Variable to set for CCS Debug mode to add extra delay for each CLI CMD
	 */
	char ccsDebugEn;

	/**
	 * @brief It Store the Module execution states
	 */
	moduleExecutionState_t gModuleExecState;

	rlDcaConfig_t  dcaConfig;
	unsigned short sensorLVDSLane;

	/******* Parameters for SPI communication ********/
	/**
	 * @brief  Advanced frame test enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char LinkAdvanceFrameTest;
	/**
	 * @brief  Continuous mode test enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char LinkContModeTest;
	/**
	 * @brief  Dynamic chirp test enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char LinkDynChirpTest;
	/**
	 * @brief  Dynamic profile test enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char LinkDynProfileTest;
	/**
	 * @brief  Advanced chirp test enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char LinkAdvChirpTest;
	/**
	 * @brief  Firmware download enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char DisableFwDownload;
	/**
	 * @brief  mmWaveLink logging enable/disable
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char EnableMmwlLogging;
	/**
	 * @brief  Calibration enable/disable
	 *         To perform calibration store/restore
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char CalibEnable;
	/**
	 * @brief  Calibration Store/Restore
	 *         If CalibEnable = 1, then whether to store/restore
	 *         1 - Store; 0 - Restore
	 */
	unsigned char CalibStoreRestore;
	/**
	 * @brief  Transport mode
	 *         1 - I2C; 0 - SPI
	 */
	unsigned char TransferMode;

	unsigned char IsFlashConnected;
	rlClientCbs_t clientCtx;
} rlDevGlobalCfg_t;

typedef struct mmwaveCommCfg
{
	/**
	 * @brief  Monitoring Report Capture enable/disable (Enable_Monitor_Capture)
	 *         1 - Enable; 0 - Disable
	 */
	unsigned char monitorCaptureEn;
	/**
	 * @brief  No. of Monitoring Report to capture.
	 */
	unsigned int totalReportToSave;
	/**
	 * @brief  mmWave Config fiel format (CONFIG_FILE_FORMAT)
	 *         1 - cfg; 0 - json
	 */
	unsigned char mmwCfgFormat;
	/**
	 * @brief  Device COM port number (COM_PORT_NUM)
	 */
	unsigned char comPortNum;
	/**
	 * @brief Store the COM port in string format including COM number
	 */
	char comPortStr[14];
	/**
	 * @brief COM Port interface handle
	 */
	HANDLE commHandle;
	/**
	 * @brief  mmwave configuration file path (CONFIG_JSON_PATH)
	 */
	char mmwConfigPath[MAX_FILE_PATH_LEN];

	/**
	 * @brief  Path to store the monitoring report in JSON format (MONITORING_JSON_PATH)
	 */
	char monReportPath[MAX_FILE_PATH_LEN];
}mmwaveCommCfg_t;


/******************************************************************************
* PARSE FUNCTION DECLARATION
******************************************************************************
*/
/* Store thr frame Trigger time in case of SPI communication (AWR1243/2243)*/
DWORD gFrameTriggerTime; 

/*get rid of trailing and leading whitespace along with "\n"*/
char *MMWL_trim(char * s);

/*Read all global variable configurations from config file*/
void MMWL_getGlobalConfigStatus(rlDevGlobalCfg_t *rlDevGlobalCfgArgs);
/* Open Configuration file in read mode */
int MMWL_openToolConfigFile();

#ifdef INTERNAL_JSON_PARSER
/* Open Configuration file in read mode */
int MMWL_openConfigFile(char *filePath, int jsonType);
int MMWL_parseJsonFile(int jsonType);
#endif
/* Close Configuration file */
void MMWL_closeConfigFile();

/*Trim the string trailing and leading whitespace*/
char *MMWL_trim(char * s);
int readCliCmdFromCfgFile(char *cliCmd);
int mmw_jsonParser(char *filePath);
void mmw_CaptureDcaDone(void);
extern void report_write(unsigned char *payload, rlUInt16_t msgId, rlUInt16_t asyncSB, int deviceType);
#endif