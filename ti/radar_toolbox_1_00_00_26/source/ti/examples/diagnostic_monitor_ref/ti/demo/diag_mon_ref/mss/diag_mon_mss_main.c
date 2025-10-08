/**
 *   @file  diag_mon_mss_main.c
 *
 *   @brief
 *      This is the main file which implements the Diagnostic and 
 *      monitoring Application
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
#include <math.h>


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
#include <ti/sysbios/family/arm/v7a/Pmu.h>
#include <ti/sysbios/family/arm/v7r/vim/Hwi.h>
#include <ti/sysbios/utils/Load.h>
#include <ti/sysbios/family/arm/exc/Exception.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include <ti/common/mmwave_sdk_version.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/esm/esm.h>
#include <ti/drivers/crc/crc.h>
#include <ti/drivers/gpio/gpio.h>
#include <ti/drivers/pinmux/pinmux.h>
#include <ti/control/dpm/dpm.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/uart/UART.h>
#include <ti/utils/mathutils/mathutils.h>


/* Demo Include Files */
#include "mss/diag_mon_mss.h"
#include "common/diag_mon_config.h"
#include "common/mmwl_if.h"
#include "common/rfmonitor.h"
#include "diag/diag_test_api.h"
#include "diag/diag_error_code.h"
#include "ti/diag/diag.h"


/* UART instance to use for Error reporting
 * 0: UART-1 for COMMAND UART.
 * 1: UART-3 for Logger UART */
#define MMW_UART_INSTANCE_ID      0
#define MMW_UART_BAUD_RATE        115200

/* Enable/disable Periodic Diagnostic Test */
#define MMW_PEIODIC_DIAG_TEST            1
/* Interval based on frame count for periodic Diagnostic test */
#define MMW_PERIODIC_DIAG_TEST_INVERVAL  100

/*! @brief   Execution of periodic Diagnostic Test event */
#define MMWDEMO_PERIODIC_DIAG_EVT                              Event_Id_03

/*! @brief   Monitor report streaming event */
#define MMWDEMO_MON_REP_STREAM_EVT                             Event_Id_04

/**
 * @brief Task Priority settings:
 */
#define MMWDEMO_MONITOR_CTRL_TASK_PRIORITY        4
#define MMWDEMO_MAIN_TASK_PRIORITY                 6

/**************************************************************************
 *************************** Global Definitions ***************************
 **************************************************************************/

/**
 * @brief
 *  Global Variable for tracking information required by the mmw Demo
 */
MmwDemo_MSS_MCB    gMmwMssMCB;
 /*Counter for number of chirps*/
 volatile uint32_t gLinkChirpCnt = 0U;

 /*Counter for number of frames*/
 volatile uint32_t gLinkFrameCnt = 0U;

 /* Store the monitoring report bits which are received from Bss */
 volatile uint32_t gmonReportData;

/**************************************************************************
 ************************* Millimeter Wave Demo Functions prototype *************
 **************************************************************************/
static void MmwDemo_initTask(UArg arg0, UArg arg1);
static void MmwDemo_platformInit(MmwDemo_platformCfg *config);
/* external sleep function when in idle (used in .cfg file) */
void MmwDemo_sleep(void);


/**************************************************************************
 ************************* Millimeter Wave Demo Functions **********************
 **************************************************************************/
/**
 *  @b Description
 *  @n
 *      Logging function which can log the messages to the CLI console
 *
 *  @param[in]  format
 *      Format string
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_CLI_write (const char* format, ...)
{
    va_list     arg;
    char        logMessage[256];
    int32_t     sizeMessage;

    /* Format the message: */
    va_start (arg, format);
    sizeMessage = vsnprintf (&logMessage[0], sizeof(logMessage), format, arg);
    va_end (arg);

    /* If CLI_write is called before CLI init has happened, return */
    if (gMmwMssMCB.commandUartHandle == NULL)
    {
        return;
    }

    /* Log the message on the UART CLI console: */
    if (gMmwMssMCB.uartPolledMode == true)
    {
        /* Polled mode: */
        UART_writePolling (gMmwMssMCB.commandUartHandle, (uint8_t*)&logMessage[0], sizeMessage);
    }
    else
    {
        /* Blocking Mode: */
        UART_write (gMmwMssMCB.commandUartHandle, (uint8_t*)&logMessage[0], sizeMessage);
    }
}

/**
 *  @b Description
 *  @n
 *      Send assert information through CLI.
 */
void _MmwDemo_debugAssert(int32_t expression, const char *file, int32_t line)
{
    if (!expression) {
        MmwDemo_CLI_write("Exception: %s, line %d.\n",file,line);
    }
}

/**
 *  @b Description
 *  @n
 *      Interrupt handler callback for frame start ISR.
 *
 *  @retval
 *      Not Applicable.
 */
void Mmwavelink_frameInterrupCallBackFunc(uintptr_t arg)
{
    /* increment Frame count */
    gLinkFrameCnt++;

    /* based on specific interval trigger the periodic Diagnostic tests */
    if((gLinkFrameCnt % MMW_PERIODIC_DIAG_TEST_INVERVAL) == 0)
    {
        /* Post event to notify frame start interrupt */
        Event_post(gMmwMssMCB.eventHandle, MMWDEMO_PERIODIC_DIAG_EVT);
    }

    if(gLinkFrameCnt > 1) //TODO THIS SHOuld be based on FTTI
    {
        Event_post(gMmwMssMCB.eventHandle, MMWDEMO_MON_REP_STREAM_EVT);
    }
}

/**
 *  @b Description
 *  @n
 *      mmw demo helper Function to start sensor.
 *
 *  @retval
 *      Success     - 0
 *  @retval
 *      Error       - <0
 */
int32_t MmwDemo_startSensor(void)
{
    int32_t     retVal;

    /*****************************************************************************
     * RF :: now start the RF and the real time ticking
     *****************************************************************************/
    retVal = MmwaveLink_frameTrigger(1);

    /*****************************************************************************
     * The sensor has been started successfully. Switch on the LED 
     *****************************************************************************/
    if(retVal == 0)
    {
        MmwDemo_CLI_write("*** Frame Triggered... ***\n");
        GPIO_write (gMmwMssMCB.cfg.platformCfg.SensorStatusGPIO, 1U);
    }

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      Stops the RF and datapath for the sensor. Blocks until both operation are completed.
 *      Prints epilog at the end.
 *
 *  @retval  None
 */
void MmwDemo_stopSensor(void)
{
    int32_t retVal;

    /* Stop sensor RF , data path will be stopped after RF stop is completed */
    retVal = MmwaveLink_frameStop(1);
    if(retVal == 0)
    {
        /* The sensor has been stopped successfully. Switch off the LED */
        GPIO_write (gMmwMssMCB.cfg.platformCfg.SensorStatusGPIO, 0U);
    }
}

/**
 *  @b Description
 *  @n
 *      Platform specific hardware initialization.
 *
 *  @param[in]  config     Platform initialization configuraiton
 *
 *  @retval
 *      Not Applicable.
 */
#if defined(SOC_XWR18XX)
#define SOC_PINN5_PADBE         SOC_XWR18XX_PINN5_PADBE
#define SOC_PINN4_PADBD         SOC_XWR18XX_PINN4_PADBD
#define SOC_PINN5_MSS_UARTA_TX  SOC_XWR18XX_PINN5_PADBE_MSS_UARTA_TX
#define SOC_PINN4_MSS_UARTA_RX  SOC_XWR18XX_PINN4_PADBD_MSS_UARTA_RX
#define SOC_PINF14_PADAJ         SOC_XWR18XX_PINF14_PADAJ
#define SOC_PINF14_MSS_UARTB_TX  SOC_XWR18XX_PINF14_PADAJ_MSS_UARTB_TX
#define SOC_PINK13_PADAZ        SOC_XWR18XX_PINK13_PADAZ
#define SOC_PINK13_GPIO2        SOC_XWR18XX_PINK13_PADAZ_GPIO_2
#define SENSOR_START_GPIO       SOC_XWR18XX_GPIO_2
#elif defined(SOC_XWR68XX)
#define SOC_PINN5_PADBE         SOC_XWR68XX_PINN5_PADBE
#define SOC_PINN4_PADBD         SOC_XWR68XX_PINN4_PADBD
#define SOC_PINN5_MSS_UARTA_TX  SOC_XWR68XX_PINN5_PADBE_MSS_UARTA_TX
#define SOC_PINN4_MSS_UARTA_RX  SOC_XWR68XX_PINN4_PADBD_MSS_UARTA_RX
#define SOC_PINF14_PADAJ         SOC_XWR68XX_PINF14_PADAJ
#define SOC_PINF14_MSS_UARTB_TX  SOC_XWR68XX_PINF14_PADAJ_MSS_UARTB_TX
#define SOC_PINK13_PADAZ        SOC_XWR68XX_PINK13_PADAZ
#define SOC_PINK13_GPIO2        SOC_XWR68XX_PINK13_PADAZ_GPIO_2
#define SENSOR_START_GPIO       SOC_XWR68XX_GPIO_2
#endif
static void MmwDemo_platformInit(MmwDemo_platformCfg *config)
{
#if (MMW_UART_INSTANCE_ID == 0)
    /* Setup the PINMUX to bring out the UART-1 */
    Pinmux_Set_OverrideCtrl(SOC_PINN5_PADBE, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(SOC_PINN5_PADBE, SOC_PINN5_MSS_UARTA_TX);
    Pinmux_Set_OverrideCtrl(SOC_PINN4_PADBD, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(SOC_PINN4_PADBD, SOC_PINN4_MSS_UARTA_RX);

#else
    /* Setup the PINMUX to bring out the UART-3 */
    Pinmux_Set_OverrideCtrl(SOC_PINF14_PADAJ, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(SOC_PINF14_PADAJ, SOC_PINF14_MSS_UARTB_TX);
#endif
    /**********************************************************************
     * Setup the PINMUX:
     * - GPIO Output: Configure pin K13 as GPIO_2 output
     **********************************************************************/
    Pinmux_Set_OverrideCtrl(SOC_PINK13_PADAZ, PINMUX_OUTEN_RETAIN_HW_CTRL, PINMUX_INPEN_RETAIN_HW_CTRL);
    Pinmux_Set_FuncSel(SOC_PINK13_PADAZ, SOC_PINK13_GPIO2);

    /**********************************************************************
     * Setup the GPIO:
     * - GPIO Output: Configure pin K13 as GPIO_2 output
     **********************************************************************/
    config->SensorStatusGPIO    = SENSOR_START_GPIO;

    /* Initialize the DEMO configuration: */
    config->sysClockFrequency   = MSS_SYS_VCLK;
    config->commandBaudRate     = MMW_UART_BAUD_RATE;

    /**********************************************************************
     * Setup the DS3 LED on the EVM connected to GPIO_2
     **********************************************************************/
    GPIO_setConfig (config->SensorStatusGPIO, GPIO_CFG_OUTPUT);
}

/**
 *  @b Description
 *  @n
 *      The function is used to synchronize mmWave on Dual Core. The
 *      function checks the status of the mmWave between the peers.
 *
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  @retval
 *      Synchronized    -   1
 *  @retval
 *      Unsynchronized  -   0
 *  @retval
 *      Error           -   <0
 */
int32_t MmwDemo_syncDss(int32_t* errCode)
{
    int32_t retVal = 0;

    //SOC_setMMWaveMSSLinkState (gMmwMssMCB.socHandle, 1, errCode);

    /* MSS: Check the status of the DSS */
    while (1)
    {
        int32_t syncStatus;
        /* Get the synchronization status: */
        syncStatus = SOC_isMMWaveDSSOperational (gMmwMssMCB.socHandle, errCode);
        if (syncStatus < 0)
        {
            return syncStatus;
        }
        if (syncStatus == 1)
        {
            /* Synchronization achieved: */
            break;
        }
        /* Sleep and poll again: */
        MmwDemo_sleep();
    }

    return retVal;
}

void DsstoMss_SW_IntHandler(uintptr_t in)
{
#ifdef SOC_XWR68XX
    MmwDemo_dssDiagTestMsg *dssDiagTestStatMsg = (MmwDemo_dssDiagTestMsg*)SOC_XWR68XX_MSS_HSRAM_BASE_ADDRESS;
#else
    MmwDemo_dssDiagTestMsg *dssDiagTestStatMsg = (MmwDemo_dssDiagTestMsg*)SOC_XWR18XX_MSS_HSRAM_BASE_ADDRESS;
#endif
    uint8_t statusBit = 0;
    uint32_t dssDiagTestStat = dssDiagTestStatMsg->diagTestBitStat,
            dssDiagTestExec = dssDiagTestStatMsg->diagTestExecBits;

    /* Diagnostic Test Status from DSS. In case of any DIAG failed at DSS, it won't proceed for next DIAG Test.
     * Print the status over UART */
    MmwDemo_CLI_write("\n*** DSS Diagnostic Status: ***\n");
    while(statusBit < DIAG_DSS_MAX_TEST_STATUS_BIT)
    {
        switch(statusBit)
        {
            case DIAG_DSS_EDMA_PARITY_C0_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS EDMA TPCC0 PARITY Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS EDMA TPCC0 PARITY Diagnostic.\n");
                }
                break;
            case DIAG_DSS_EDMA_PARITY_C1_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS EDMA TPCC1 PARITY Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS EDMA TPCC1 PARITY Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L1P_PARITY_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L1P PARITY Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L1P PARITY Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L2P_PARITY_P0_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L2P UMAP0 PARITY Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L2P UMAP0 PARITY Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L2P_PARITY_P1_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L2P UMAP1 PARITY Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L2P UMAP1 PARITY Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L2_ECC_1B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L2 ECC 1Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L2 ECC 1Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L2_ECC_2B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L2 ECC 2Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L2 ECC 2Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L3_ECC_1B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L3 ECC 1Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L3 ECC 1Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_L3_ECC_2B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS L3 ECC 2Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS L3 ECC 2Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_TXFR_RAM_ECC_1B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS TXFR RAM ECC 1Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS TXFR RAM ECC 1Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_TXFR_RAM_ECC_2B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS TXFR RAM ECC 2Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS TXFR RAM ECC 2Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_HSRAM_ECC_1B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS HSRAM ECC 1Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS HSRAM ECC 1Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_HSRAM_ECC_2B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS HSRAM ECC 2Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS HSRAM ECC 2Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_HWA_ECC_1B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS HWA ECC 1Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS HWA ECC 1Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_HWA_ECC_2B_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS HWA ECC 2Bit Error Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS HWA ECC 2Bit Error Diagnostic.\n");
                }
                break;
            case DIAG_DSS_HWA_LOCKSTEP_TEST_STATUS_BIT:
                /* If this test has been executed and failed */
                if((dssDiagTestExec & (1<<statusBit)) && (dssDiagTestStat & (1<<statusBit)))
                {
                    MmwDemo_CLI_write("[ERROR] DSS HWA Lockstep Diagnostic Failed [%d]\n",\
                                      dssDiagTestStatMsg->errVal);
                }
                else
                {
                    MmwDemo_CLI_write("[SUCCESS] DSS HWA Lockstep Error Diagnostic.\n");
                }
                break;
            }
        statusBit++;
        }
}

/**
 *  @b Description
 *  @n
 *      The function is transmit failure monitor and execute the periodic Diagnostic test.
 *
 *  @param[out] errCode
 *      Error code populated by the API on an error
 *
 *  @retval None
 */
static void MmwDemo_MonReportStreamTask (UArg arg0, UArg arg1)
{
    unsigned int matchingEvents;

    /* Execute forever: */
    while (1)
    {
        matchingEvents = Event_pend(gMmwMssMCB.eventHandle, Event_Id_NONE, \
                                    MMWDEMO_MON_REP_STREAM_EVT | MMWDEMO_PERIODIC_DIAG_EVT, BIOS_WAIT_FOREVER);
       if(matchingEvents != 0)
       {
           /* Execute Periodic Diag Test at fixed interval */
           if((matchingEvents & MMWDEMO_PERIODIC_DIAG_EVT) && MMW_PEIODIC_DIAG_TEST)
           {
               Diag_StaticCfgErrInfo errorInfo = {0};
               int32_t               retVal;

               MmwDemo_CLI_write("*** PERIODIC Static Diagnostic Test ***\n");

               /* ESM Static config test */
               retVal = MssDiag_EsmStaticTest(&errorInfo, NULL);

               /* VIM Static config test */
               retVal = MssDiag_vimStaticTest(&errorInfo, NULL);

               /* DMA Static test : instance ID 0 */
               retVal = MssDiag_dmaStaticTest(0U, &errorInfo, NULL);

               /* RCM Static Configuration Diagnostic Test */
               retVal = MssDiag_RCMStaticTest(&errorInfo, NULL);

               /* R4F Static Configuration Diagnostic Test */
               retVal = MssDiag_r4fStaticTest(&errorInfo, NULL);

               /* RTI Static configuration Diagnostic test */
               retVal = MssDiag_RtiStaticCfgTest(&errorInfo, NULL);

               MMW_MSS_UNUSED_VAR(retVal);
           }
           else if(matchingEvents & MMWDEMO_MON_REP_STREAM_EVT)
           {
               /* In this application ONLY Failure Monitor reports are reported
                * MmwaveLink_asyncEventHandler -> RFMon_reportHandler ->RFMon_reportFailureToHost
                */
           }
           else
           {
               /* Do nothing */
           }
       }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is to configuration communication interfaces.
 *
 *  @param[out] None
 *
 *  @retval None
 */
void MmwDemo_commInterfaceInit(void)
{
    UART_Params         uartParams;

    /*****************************************************************************
     * Initialize the mmWave SDK components:
     *****************************************************************************/
    /* Initialize the UART */
    UART_init();

    /* Initialize the GPIO */
    GPIO_init();

    /* Platform specific configuration */
    MmwDemo_platformInit(&gMmwMssMCB.cfg.platformCfg);

    /*****************************************************************************
     * Open the mmWave SDK components:
     *****************************************************************************/
    /* Setup the default UART Parameters */
    UART_Params_init(&uartParams);
    uartParams.clockFrequency = gMmwMssMCB.cfg.platformCfg.sysClockFrequency;
    uartParams.baudRate       = gMmwMssMCB.cfg.platformCfg.commandBaudRate;
    uartParams.isPinMuxDone   = 1;
    /* set UART as polling mode */
    gMmwMssMCB.uartPolledMode  = true;

    /* Open the UART Instance */
    gMmwMssMCB.commandUartHandle = UART_open(MMW_UART_INSTANCE_ID, &uartParams);
    if (gMmwMssMCB.commandUartHandle == NULL)
    {
        MmwDemo_debugAssert (0);
        return;
    }
}

volatile int debugTag = 0;

/**
 *  @b Description
 *  @n
 *      System Initialization Task which initializes the various
 *      components in the system.
 *
 *  @retval
 *      Not Applicable.
 */
static void MmwDemo_initTask(UArg arg0, UArg arg1)
{
    int32_t             errCode, retVal;
    Task_Params         taskParams;

    /* Debug Message: */
    System_printf("Debug: Launched the Initialization Task\n");

    /*****************************************************************************
     *  UART Loopback test both of instances before those are being used
    *****************************************************************************/
    retVal = MssDiag_UartLoopbackTest(0, NULL);
    retVal = MssDiag_UartLoopbackTest(1, NULL);

    /* initialize the Communication interface like UART & GPIO */
    MmwDemo_commInterfaceInit();

    /* Print the application welcome banner over UART */
    MmwDemo_CLI_write ("**********************************************\n");
    MmwDemo_CLI_write ("     mmWave Safety Diagnostic Library Demo \n");
    MmwDemo_CLI_write ("**********************************************\n");

    /*****************************************************************
     * Boot up test verification:
     *  1. Bootloader test status is already being verified by the SBL,
     *     provided here for reference purpose.
     *  2. This application is booting that proves that PBIST & STC
     *     diagnostic tests are verified SUCCESSFULLY by the SBL.
     *****************************************************************/
    retVal = MssDiag_bootTestStatus(NULL);

    /*****************************************************************************
     * Execute all the Error Injection Tests as part of initial diagnostic test at
     * the application and print the failure result over UART.
     */
    retVal = MssDiag_ErrorInjectTest();
    
    MssDiag_SelfTest();
    
    MssDiag_StaticConfigTest();

    /*****************************************************************************
     * By this time DSS core and application is already up.
     * Synchronization: This will synchronize the execution of the control module
     * between the domains. This is a prerequiste and always needs to be invoked.
     *****************************************************************************/
    if (MmwDemo_syncDss(&errCode) < 0)
    {
        /* Error: Unable to synchronize the mmWave control module */
        System_printf ("Error: DSS Synchronization failed [Error code %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return;
    }
    System_printf ("Debug: DSS Synchronization was successful\n");

    /*****************************************************************************
     * Mailbox ECC Error Injection Diagnostic test has been done at the beginning
     * of this application as part of app-boot time error injection tests.
     *****************************************************************************/
    /* Initialize the Mailbox */
    Mailbox_init(MAILBOX_TYPE_MSS);

    /* Default instance configuration params */
    gMmwMssMCB.eventHandle = Event_create(NULL, 0);
    if (gMmwMssMCB.eventHandle == NULL)
    {
        /* FATAL_TBA */
        printf("Error: MMWDemoMSS Unable to create an event handle\n");
        return ;
    }

    /* Set the MSS link status so DSS core can get to know of MSS state. */
    SOC_setMMWaveMSSLinkState (gMmwMssMCB.socHandle, 1, &errCode);

    /*****************************************************************************
     * Setup and initialize the mmWave Link:
     *****************************************************************************/
    if (MmwaveLink_initLink (RL_AR_DEVICETYPE_18XX, RL_PLATFORM_MSS ) < 0)
    {
        System_printf ("mmWave Link Initialization failed");
        return;
    }
    /* pass CLI function pointer to MMWL_IF */
    MmwaveLink_setLogFunc(MmwDemo_CLI_write);
    
    /*****************************************************************************
     * Launch the Monitor report task.
     * It will report Failure monitors over UART.
     *****************************************************************************/
    Task_Params_init(&taskParams);
    taskParams.priority  = MMWDEMO_MONITOR_CTRL_TASK_PRIORITY;
    taskParams.stackSize = 3*1024;
    gMmwMssMCB.taskHandles.monReportTask = Task_create(MmwDemo_MonReportStreamTask, &taskParams, NULL);

    retVal = MmwaveLink_Config();
    MmwDemo_debugAssert(retVal == 0);

    /* Initialize the monitoring structures to the predefined configuration */
    RFMon_initCfg();

    /* invoke Monitoring APIs here */
    retVal = RFMon_config(&errCode);
    MmwDemo_debugAssert(retVal == 0);

    retVal = MmwDemo_startSensor();
    MmwDemo_debugAssert(retVal == 0);

    return;
}

/**
 *  @b Description
 *  @n
 *     Function to sleep the R4F using WFI (Wait For Interrupt) instruction.
 *     When R4F has no work left to do,
 *     the BIOS will be in Idle thread and will call this function. The R4F will
 *     wake-up on any interrupt (e.g chirp interrupt).
 *
 *  @retval
 *      Not Applicable.
 */
void MmwDemo_sleep(void)
{
    /* issue WFI (Wait For Interrupt) instruction */
    asm(" WFI ");
}

/**
 *  @b Description
 *  @n
 *      Entry point into the Millimeter Wave Demo
 *
 *  @retval
 *      Not Applicable.
 */
int main (void)
{
    Task_Params     taskParams;
    int32_t         errCode;
    SOC_Handle      socHandle;
    SOC_Cfg         socCfg;
    SOC_SysIntListenerCfg    linkFrameCfg;
    HwiP_Params         hwiParams;

    /* Initialize the ESM: Don't clear errors as TI RTOS does it
      OSAL ESM driver internally calls mmwave SDK ESM driver init function */
    MssDiag_IntEsmDrvInit();

    /* Initialize the SOC configuration: */
    memset ((void *)&socCfg, 0, sizeof(SOC_Cfg));

    /* Populate the SOC configuration: */
    socCfg.clockCfg = SOC_SysClock_INIT;
    socCfg.mpuCfg = SOC_MPUCfg_BYPASS_CONFIG;
    socCfg.dssCfg = SOC_DSSCfg_UNHALT;

    /* Initialize the SOC Module: This is done as soon as the application is started
     * to ensure that the MPU is correctly configured. */
    socHandle = SOC_init (&socCfg, &errCode);
    if (socHandle == NULL)
    {
        System_printf ("Error: SOC Module Initialization failed [Error code %d]\n", errCode);
        MmwDemo_debugAssert (0);
        return -1;
    }    

    /* Register frame interrupt handler.
     * This interrupt is being used for
     * 1. periodic monitor report processing
     * 2. Run periodic DIAG tests
     * 3. External WDT sync (if available)
     */
    memset((void *)&linkFrameCfg, 0 , sizeof(SOC_SysIntListenerCfg));
    linkFrameCfg.systemInterrupt    = MMW_MSS_FRAME_START_INT;
    linkFrameCfg.listenerFxn        = Mmwavelink_frameInterrupCallBackFunc;
    linkFrameCfg.arg                = (uintptr_t)NULL;
    if ((SOC_registerSysIntListener(socHandle, &linkFrameCfg, &errCode)) == NULL)
    {
        System_printf("Error: Unable to register frame interrupt listener , error = %d\n", errCode);
        return -1;
    }

    /* Check if the SOC is a secure device.
     * NOTE: This application is not being tested on secure device. */
    if (SOC_isSecureDevice(socHandle, &errCode))
    {
        /* Disable firewall for JTAG and LOGGER (UART) which is needed by all unit tests */
        SOC_controlSecureFirewall(socHandle, 
                                  (uint32_t)(SOC_SECURE_FIREWALL_JTAG | SOC_SECURE_FIREWALL_LOGGER),
                                  SOC_SECURE_FIREWALL_DISABLE,
                                  &errCode);
    }

    /* Initialize and populate the demo MCB */
    memset ((void*)&gMmwMssMCB, 0, sizeof(MmwDemo_MSS_MCB));

    gMmwMssMCB.socHandle = socHandle;

    HwiP_Params_init(&hwiParams);
    hwiParams.name = "DSS_MSS_SW";
    hwiParams.arg  = NULL;
#if defined(SOC_XWR18XX)
    HwiP_create(SOC_XWR18XX_MSS_DSS2MSS_SW1_INT, &DsstoMss_SW_IntHandler, &hwiParams);
#else
    HwiP_create(SOC_XWR68XX_MSS_DSS2MSS_SW1_INT, &DsstoMss_SW_IntHandler, &hwiParams);
#endif
    /* Debug Message: */
    System_printf ("**********************************************\n");
    System_printf ("  Launching the DIAG & MONITOR Demo on MSS\n");
    System_printf ("**********************************************\n");

    /* Initialize the Task Parameters. */
    Task_Params_init(&taskParams);
    taskParams.stackSize = 10*1024;
    taskParams.priority = MMWDEMO_MAIN_TASK_PRIORITY;
    gMmwMssMCB.taskHandles.initTask = Task_create(MmwDemo_initTask, &taskParams, NULL);

    /* Start BIOS */
    BIOS_start();
    return 0;
}


