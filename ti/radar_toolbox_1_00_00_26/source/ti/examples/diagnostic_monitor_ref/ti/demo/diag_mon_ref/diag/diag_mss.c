/**
 *   @file  diag_mss.c
 *
 *   @brief
 *     This file contains all the SDL tests need to execute on MSS core.
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
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <xdc/runtime/System.h>

/* mmWave SDL Include Files. */
#include <ti/csl/csl.h>
#include <ti/diag/diag.h>
#include <ti/diag/include/Diag_internal.h>
#include "osal/osal.h"
#include <ti/common/sys_common.h>
#include "diag/diag_test_api.h"
/* Diagnostic Error codes */
#include "diag/diag_error_code.h"

#define UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(a, b)  {a = ((b != NULL) ? (a + *b) : a);}
#define UPDATE_DIAG_TEST_CYCLE_COUNT(a, b)      if(a!=NULL){*a += b;}

#define MMW_PRINT       MmwDemo_CLI_write

#define ASSERT(a)  if(!a){while(0);}
/* for xWR1843 device variant */
#ifdef SOC_XWR18XX
#define SOC_MSS_ESM_LOW_PRIORITY_INT   SOC_XWR18XX_MSS_ESM_LOW_PRIORITY_INT
#endif
/* for xWR684x device variant */
#ifdef SOC_XWR68XX
#define SOC_MSS_ESM_LOW_PRIORITY_INT   SOC_XWR68XX_MSS_ESM_LOW_PRIORITY_INT
#endif
/* MPU region Index for DMA MPU Error Injection Diagnostic Test.
 * Range : 0 to 3.
 * It's fixed one region now but user can change it to range of MPU regions */
#define DMA_MPU_REGION_IDX                2U
#define DMA_ERR_INJECTION_START_MPU_IDX  DMA_MPU_REGION_IDX
#define DMA_ERR_INJECTION_END_MPU_IDX    DMA_MPU_REGION_IDX

/* DMA Channel Index for Parity Error Injection Diagnostic Test.
 * Range : 0 to 31.
 * It's fixed one channel now but user can change it to range of DMA Channel */
#define DMA_ERR_INJECT_CHAN               4U
#define DMA_ERR_INJECTION_START_CHAN_IDX  DMA_ERR_INJECT_CHAN
#define DMA_ERR_INJECTION_END_CHAN_IDX    DMA_ERR_INJECT_CHAN

/* MibSPI RAM location index for ECC Error Injection Diagnostic Test.
 * Range : 0 to 127.
 * It's fixed one location now but user can change it to range of RAM location  */
#define MIBSPI_RAM_BUFFER_IDX_ECC_INJECT         10
#define MIBSPI_RAM_ECC_START_IDX                 MIBSPI_RAM_BUFFER_IDX_ECC_INJECT
#define MIBSPI_RAM_ECC_END_IDX                   MIBSPI_RAM_BUFFER_IDX_ECC_INJECT


/* CCCA Pre-calculated values for test */
/*
 * Clk1 freq has to be GREATER than Clk0 freq for correct functionality
 * Clock0 Freq (fClock0) = 40MHz
 * Clock1 Freq (fClock1) = 240MHz
 * Desired freq accuracy = 0.1%
 *
 * Calculation:
 * TClock0=1/40MHz= 25ns
 * TClock1=1/240MHz=4.167ns
 * MARGINMIN=(synchronization+digitization)= (2+2)*(TClock0/TClock1)=4*(25/4.167)=24
 * Freq accuracy of 0.1% translates to 1/1000=1/RESOLUTION
 * DURATIONMIN = MARGINMIN*RESOLUTION/fClock1 = 24*1000/(240x10e6)=.0001 (0.1ms)
 * DURATIONMIN = TClock0 * Counter0 = TClock1 * Counter1
 *   .0001 = .000000025 * Counter0
 *   Counter0 = 0.0001/.000000025 = 4000 (Counter0 expiry)
 *
 *   .0001 = .000000004167 * Counter1
 *   Counter1 = 0.0001/.000000004167 = 24000 (Counter1 expected)
 *
 *   Timeout count > Duration = Counter 1 + Counter1/4 (25% additional time to complete comparison)
 *                            = 24000 + 24000/4 = 30000 (Count1 error)
 */
#define CCCA_CLK_SRC0               Diag_CLKSRC_REF_CLK     /* 40MHz clock source for counter 0 */
#define CCCA_CLK_SRC1               Diag_CLKSRC_PLL_240_CLK /* 240MHz clock source for counter 1 */
#define CCCA_COUNT0_EXPIRY          4000U
#define CCCA_COUNT1_EXPECTED        24000U
#define CCCA_MARGIN_COUNT           24U
#define CCCA_COUNT1_ERROR           30000U

/* CCCB Pre-calculated values for test */
/*
 * Clk1 freq has to be GREATER than Clk0 freq for correct functionality
 * Clock0 Freq (fClock0) = 40MHz
 * Clock1 Freq (fClock1) = 200MHz
 * Desired freq accuracy = 0.1%
 *
 * Calculation:
 * TClock0=1/40MHz= 25ns
 * TClock1=1/200MHz=5ns
 * MARGINMIN=(synchronization+digitization)= (2+2)*(TClock0/TClock1)=4*(25/5)=20
 * Freq accuracy of 0.1% translates to 1/1000=1/RESOLUTION
 * DURATIONMIN = MARGINMIN*RESOLUTION/fClock1 = 20*1000/(200x10e6)=.0001 (0.1ms)
 * DURATIONMIN = TClock0 * Counter0 = TClock1 * Counter1
 *   .0001 = .000000025 * Counter0
 *   Counter0 = 0.0001/.000000025 = 4000 (Counter0 expiry)
 *
 *   .0001 = .000000005 * Counter1
 *   Counter1 = 0.0001/.000000005 = 20000 (Counter1 expected)
 *
 *   Timeout count > Duration = Counter1 + Counter1/4 (25% additional time to complete comparison)
 *                            = 20000 + 20000/4 = 25000 (Count1 error)
 */
#define CCCB_CLK_SRC0               Diag_CLKSRC_REF_CLK  /* 40MHz clock source for counter 0 */
#define CCCB_CLK_SRC1               Diag_CLKSRC_BSS_CLK  /* 200MHz clock source for counter 1 */
#define CCCB_COUNT0_EXPIRY          4000U
#define CCCB_COUNT1_EXPECTED        20000U
#define CCCB_MARGIN_COUNT           20U
#define CCCB_COUNT1_ERROR           25000U

/* Loop count used to stress test the core CCM tests */
#define CCM_STRESS_LOOP_COUNT 20

/* The poll timeout value is a conservative number that takes into account
 * normal tests as well as test using instrumented code which takes longer.
 * An application should calculate the MAX_CYCLE_COUNT with the expected test
 * run time.
 * In the above 2 cases the test duration is 0.1ms which translates to
 * 20000 cycles at 200MHz MSS clk. Use * 2 factor gives 40000.
 */
#define CCC_MAX_CYCLE_COUNT 40000U

/* In continuous mode pass case a timeout is expected since there is no Done condition */
#define CCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE 50000000U   /* 250msec at 200MHz */


/* DCCA Pre-calculated values for test (refer to Safety manual for details) */
/*
 * Clock0 Freq (fClock0) = 240MHz
 * Clock1 Freq (fClock1) = 40MHz
 * Desired freq accuracy = 0.1%
 *
 * Calculation:
 * TClock0=1/240MHz=4.167ns
 * TClock1=1/40MHz= 25ns
 * ValidMin = 2 * (synchronization+digitization)
 *          = 2 * (2x(TClock1/TClock0) + 3) if TClock1 > TClock0
 *          = 2 * (2*(25/4.167) + 3) = 2 * ( 2 * 6 + 3)
 *          = 30  (ValidSeed0)
 * Freq accuracy of 0.1% translates to 1/1000=1/RESOLUTION
 * DurationMin = ValidMin * RESOLUTION / (2 * fClock0)
 *             = 30 * 1000/(2 * 240x10e6) = .0000625s = 62.5uS
 *
 * DurationMin = TClock0 * CountSeed0 = TClock1 * CountSeed1
 *
 *   0.0000625  = 0.000000004167 * CountSeed0
 *   CountSeed0 = 0.0000625/0.000000004167 = 15000
 *   CountSeed0 adjusted = CountSeed0 - (ValidMin/2)
 *                     = 15000 - 30/2
 *                     = 14985 (CountSeed0)
 *
 *   0.0000625  = 0.000000025 * CountSeed1
 *   CountSeed1 = 0.0000625/0.000000025 = 2500 (CountSeed1)
 *
 */
#define DCCA_CLK_SRC0       Diag_CLKSRC_PLL_240_CLK
#define DCCA_CLK_SRC1       Diag_CLKSRC_REF_CLK
#define DCCA_COUNTSEED0     14985U
#define DCCA_COUNTSEED1     2500U
#define DCCA_VALIDSEED0     30U

/* DCCB Pre-calculated values for test (refer to Safety manual for details) */
/*
 * Clock0 Freq (fClock0) = 40MHz
 * Clock1 Freq (fClock1) = 200MHz
 * Desired freq accuracy = 0.1%
 *
 * Calculation:
 * TClock0=1/40MHz= 25ns
 * TClock1=1/200MHz=5ns
 * ValidMin = 2 * (synchronization+digitization)
 *          = 2 * (2+3) for TClock0 > TClock1
 *          = 2 * (5)
 *          = 10 (ValidSeed0)
 * Freq accuracy of 0.1% translates to 1/1000=1/RESOLUTION
 * DurationMin = ValidMin * RESOLUTION / (2 * fClock0)
 *             = 10 * 1000/(2 * 40x10e6) = .000125s = 125uS
 *
 * DurationMin = TClock0 * CountSeed0 = TClock1 * CountSeed1
 *
 *   0.000125 = 0.000000025 * CountSeed0
 *   CountSeed0 = 0.000125/0.000000025 = 5000
 *   CountSeed0 adjusted = CountSeed0 - (ValidMin/2)
 *                     = 5000 - 10/2
 *                     = 4995 (CountSeed0)
 *
 *   0.000125 = 0.000000005 * CountSeed1
 *   CountSeed1 = 0.000125/0.000000005 = 25000 (CountSeed1)
 *
 */
#define DCCB_CLK_SRC0       Diag_CLKSRC_CPU_CLK
#define DCCB_CLK_SRC1       Diag_CLKSRC_VCLK
#define DCCB_COUNTSEED0     4995U
#define DCCB_COUNTSEED1     25000U
#define DCCB_VALIDSEED0     10U

/* The timeout value is a conservative number that takes into account
 * normal tests as well as test using instrumented code which takes longer.
 * An application should calculate the MAX_CYCLE_COUNT with the expected test
 * run time.
 */
#define DCC_MAX_CYCLE_COUNT 50000U
/* In continuous mode pass case a timeout is expected since there is no Done condition */
#define DCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE 500000U

/**
 * @brief
 *  This is the TCMA memory region (PROG_RAM)
 */
#pragma DATA_SECTION (gTCMAMemoryRegion, ".tcma");
uint32_t gTCMAMemoryRegion;

/**
 * @brief
 *  This is the TCMB memory region (DATA_RAM)
 */
#pragma DATA_SECTION (gTCMBMemoryRegion, ".tcmb");
uint32_t gTCMBMemoryRegion;

/**
 * @brief
 *  This is the L3 memory region
 */
#pragma DATA_SECTION (gL3MemoryRegion, ".l3ram");
uint32_t gL3MemoryRegion;

/**
 * @brief
 *  This is the HSRAM memory region
 */
#pragma DATA_SECTION (gHSRAMMemoryRegion, ".hsram");
uint32_t gHSRAMMemoryRegion;


#define NUM_MAILBOX_MEMORY_TYPE 6U
/* Timeout for all mailbox ECC tests */
#define MAILBOX_ECC_MAX_CYCLE_COUNT 10000U

CSL_RCM_MemoryType mailboxMemoryType[NUM_MAILBOX_MEMORY_TYPE] =
{
    CSL_RCM_MemoryType_MAILBOX_MSS_TO_BSS,
    CSL_RCM_MemoryType_MAILBOX_BSS_TO_MSS,
    CSL_RCM_MemoryType_MAILBOX_DSS_TO_BSS,
    CSL_RCM_MemoryType_MAILBOX_DSS_TO_MSS,
    CSL_RCM_MemoryType_MAILBOX_MSS_TO_DSS,
    CSL_RCM_MemoryType_MAILBOX_BSS_TO_DSS
};

const char * mailboxMemoryTypeName[NUM_MAILBOX_MEMORY_TYPE] =
{
    "MSS to BSS",
    "BSS to MSS",
    "DSS to BSS",
    "DSS to MSS",
    "MSS to DSS",
    "BSS to DSS"
};

const char * tcmMemoryTypeName[3] =
{
    "TCMA ",
    "TCMB0",
    "TCMB1"
};

typedef struct diagErrorDetail
{
    int32_t   errCode;

    uint64_t  errCauseInfo;

}diagErrDetail_t;


diagErrDetail_t gDiagErrDetail;

/* MSS Diagnostic Test Status Bit
 * This stores the Diagnostic test status at each bits */
uint64_t gMssDiagTestStatus = 0;

extern void OSAL_R4F_InterruptInit(void);
extern void OSAL_R4F_ESMDrv_init(void);
extern void OSAL_R4F_CycleProfiler_init(uint32_t pmuCounterId);
extern void MmwDemo_CLI_write (const char* format, ...);

/**
 *  @b Description
 *  @n
 *      Funtion to initialize the R4F Interrupt, ESM and Cycle profiler
 *
 *  @retval  None
 */
void MssDiag_IntEsmDrvInit()
{
    /* Initialize the OSAL Interrupt Module. */
    OSAL_R4F_InterruptInit();

    /* Initialize the OSAL Cycle Profiling Module. */
    OSAL_R4F_CycleProfiler_init(1U);

    /* Initialize the OSAL ESM Driver. */
    OSAL_R4F_ESMDrv_init();

}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the UART IO peripheral diagnostic. It allows
 *      enabling and disabling of loopback mode.
 *
 *  @param[in] instanceId  UART Instance ID
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_UartLoopbackTest(uint8_t instanceId, uint32_t* ptrCycleCount)
{

    Diag_UART_LoopbackCfgParams params;
    CSL_SCIRegs*                ptrUARTRegs;
    uint32_t                    cycleCount = 0;
    int32_t                     retVal;

    /* Get the base address of the UART Module */
    ptrUARTRegs = CSL_UART_getBaseAddress(instanceId);

    /* Reset the module */
    CSL_UART_setResetStatus(ptrUARTRegs, 0U);

    /* Bring the module out of reset */
    CSL_UART_setResetStatus(ptrUARTRegs, 1U);

    /***************************************************************************
     * Diagnostic: UART Instance 0/1 Loopback Mode Disabled
     ***************************************************************************/
    /* Populate the configuration. */
    (void)memset((void *)&params, 0, sizeof(Diag_UART_LoopbackCfgParams));

    /* Populate the parameters. */
    params.instanceId   = instanceId;
    params.mode         = 0U;

    /* Execute the diagnostic. */
    retVal = Diag_UART_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag UART loopback [Instance %d Mode Disabled] \n", \
                instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /*********************************************************************
     * Diagnostic: UART Instance 0/1 Loopback Mode Enabled
     ***************************************************************************/
    /* Populate the configuration. */
    (void)memset((void *)&params, 0, sizeof(Diag_UART_LoopbackCfgParams));

    /* Populate the parameters. */
    params.instanceId   = instanceId;
    params.mode         = 1U;

    /* Execute the diagnostic. */
    retVal = Diag_UART_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag UART loopback [Instance %d Mode Enabled] \n", \
                instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag UART-%d Loopback Test \n", instanceId);
EXIT:
    if(instanceId == 0)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_UART0_LOOPBACK_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_UART1_LOOPBACK_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to test the MIBSPI loopback Diagnostic
 *      with basic configuration. The results of the test execution
 *      and the time taken to execute the diagnostic are logged into
 *      cycleCount.
 *
 *  @param[in] instanceID     MibSPI Instance ID
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_MibSPILoopbackTest (uint8_t instanceId, uint32_t* ptrCycleCount)
{
    Diag_MIBSPI_LoopbackCfgParams   params;
    CSL_MIBSPIRegs*                 ptrMIBSPIRegs;
    uint8_t                         done;
    uint32_t                        cycleCount = 0;
    int32_t                         retVal;

    /* Get the base address of the MIBSPI Module */
    ptrMIBSPIRegs = CSL_MIBSPI_getBaseAddress(instanceId);

    /* Reset the module */
    CSL_MIBSPI_reset(ptrMIBSPIRegs);

    /* Wait for the Multibuffer RAM initialization to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_MIBSPI_getMultiBufferInitStatus(ptrMIBSPIRegs) == 0U)
        {
            /* YES: We can exit the loop. */
            done = 1U;
        }
    }
    /***************************************************************************
     * Diagnostic: MIBSPI Instance 0/1 Loopback Mode Disabled
     ***************************************************************************/
    /* Populate the configuration. */
    params.instanceId   = instanceId;
    params.mode         = Diag_MIBSPI_LoopbackType_DISABLED;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag MibSPI loopback [Instance %d Mode Disabled] \n", \
                instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: MIBSPI Instance 0/1 Loopback Mode Internal
     ***************************************************************************/
    /* Populate the configuration. */
    params.instanceId   = instanceId;
    params.mode         = Diag_MIBSPI_LoopbackType_INT;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag MibSPI loopback [Instance %d Mode Internal] \n", \
                instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: MIBSPI Instance 0/1 Loopback Mode Analog
     ***************************************************************************/
    /* Populate the configuration. */
    params.instanceId   = instanceId;
    params.mode         = Diag_MIBSPI_LoopbackType_IO_ANALOG;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag MibSPI loopback [Instance %d Mode Analog] \n", \
                instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: MIBSPI Instance 0/1 Loopback Mode Digital
     ***************************************************************************/
    /* Populate the configuration. */
    params.instanceId   = 0U;
    params.mode         = Diag_MIBSPI_LoopbackType_IO_DIGITAL;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_loopbackExecute(&params, NULL);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag MibSPI loopback [Instance %d Mode Digital] \n", \
                instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag MibSPi-%d Loopback Test \n", instanceId);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_MIBSPI0_LOOPBACK_TEST_STATUS_BIT+instanceId));
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to test the MCAN loopback Diagnostic
 *      with basic configuration. The results of the test execution
 *      and the time taken to execute the diagnostic are logged into
 *      cycleCount.
 *
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *
 */
int32_t MssDiag_McanLoopbackTest(uint32_t* ptrCycleCount)
{
    CSL_MCANRegs*               ptrMCANRegs;
    uint8_t                     done;
    Diag_MCAN_LoopbackCfgParams params;
    uint32_t                    cycleCount = 0;
    int32_t                     retVal;

    /******************************************************************
     * MCAN Module Initialization:
     ******************************************************************/

    /* Get the base address of the MCAN Module */
    ptrMCANRegs = CSL_MCAN_getBaseAddress();

    /* Reset the module */
    CSL_MCAN_reset(ptrMCANRegs);

    /* Wait for the MCAN module reset to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is reset done? */
        if (CSL_MCAN_getResetStatus(ptrMCANRegs) == 0U)
        {
            /* YES: We can exit the loop. */
            done = 1U;
        }
    }
    /***************************************************************************
     * Diagnostic: MCAN Loopback Mode Disabled
     ***************************************************************************/
    /* Populate the configuration. */
    (void)memset((void *)&params, 0, sizeof(Diag_MCAN_LoopbackCfgParams));

    /* Populate the parameters. */
    params.maxCycleCount    = 1000U;
    params.mode             = 0U;

    /* Execute the diagnostic. */
    retVal = Diag_MCAN_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag MCAN loopback [Mode Disabled] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: MCAN Loopback Mode Enabled
     ***************************************************************************/
    /* Populate the configuration. */
    (void)memset((void *)&params, 0, sizeof(Diag_MCAN_LoopbackCfgParams));

    /* Populate the parameters. */
    params.maxCycleCount    = 1000U;
    params.mode             = 1U;

    /* Execute the diagnostic. */
    retVal = Diag_MCAN_loopbackExecute(&params, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag MCAN loopback [Mode Enabled] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag MCAN Loopback \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MCAN_LOOPBACK_TEST_STATUS_BIT);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to test the I2C loopback Diagnostic
 *      with basic configuration. The results of the test execution
 *      and the time taken to execute the diagnostic are logged into
 *      cycleCount
 *
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *
 */
int32_t MssDiag_I2cLoopbackTest(uint32_t* ptrCycleCount)
{
    uint8_t         mode;
    uint32_t        cycleCount;
    int32_t         retVal;

    /***************************************************************************
     * Diagnostic: I2C Loopback Mode Disabled
     ***************************************************************************/
    /* Populate the configuration. */
    mode = 0U;

    /* Execute the diagnostic. */
    retVal = Diag_I2C_loopbackExecute(mode, &cycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag I2C loopback [Mode Disabled] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: I2C Loopback Mode Enabled
     ***************************************************************************/
    /* Populate the configuration. */
    mode = 1U;

    /* Execute the diagnostic. */
    retVal = Diag_I2C_loopbackExecute(mode, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag I2C loopback [Mode Enabled] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag I2C Loopback \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_I2C_LOOPBACK_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the RTI Static configuration
 *      diagnostic.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_RtiStaticCfgTest (Diag_StaticCfgErrInfo *ptrErrInfo,
                                          uint32_t *ptrCycleCount)
{
    CSL_RCMRegs*                ptrRCMRegs;
    int32_t                     retVal;
    Diag_RTI_StaticCfgParams    params;
    uint32_t                    cycleCount = 0;

    /* Get the base address of RCM Module */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    /* Enable the Watchdog functionality. */
    CSL_RCM_setWatchdogStatus(ptrRCMRegs, 1U);

    /********************************************************************
     * Test:
     * - Initialization Cycle count
     ********************************************************************/
    (void)memset ((void *)&params, 0, sizeof(Diag_RTI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = 1U;
    params.mode         = Diag_StaticCfgMode_INIT;

    /* Execute the diagnostic. */
    retVal = Diag_RTI_staticCfgExecute (&params, ptrErrInfo, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag RTI Static Configuration [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Test:
     * - Compare Cycle count
     ********************************************************************/
    (void)memset ((void *)&params, 0, sizeof(Diag_RTI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = 1U;
    params.mode         = Diag_StaticCfgMode_COMPARE;

    /* Execute the diagnostic. */
    retVal = Diag_RTI_staticCfgExecute (&params, ptrErrInfo, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag RTI Static Configuration [Compare] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Test:
     * - Reset Cycle count
     ********************************************************************/
    (void)memset ((void *)&params, 0, sizeof(Diag_RTI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = 1U;
    params.mode         = Diag_StaticCfgMode_RESET;

    /* Execute the diagnostic. */
    retVal = Diag_RTI_staticCfgExecute (&params, ptrErrInfo, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag RTI Static Configuration [Reset] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag RTI Static Config \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_RTI_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the Watchdog diagnostic. It enables the
 *      watchdog with a preload count of 0. The watchdog is not serviced hence
 *      it timesout and generates a NMI.
 *
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_WatchdogTest (uint32_t* ptrCycleCount)
{
    int32_t                 retVal;
    uint32_t                maxCycleCount;

    /***************************************************************************
     * Diagnostic: Watchdog NMI
     ***************************************************************************/
    /* Populate the configuration. */
    maxCycleCount = 10000U;

    /* Execute the diagnostic. */
    retVal = Diag_Watchdog_execute(maxCycleCount, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag Watchdog Test\n", retVal);
        goto EXIT;
    }
    MMW_PRINT("[SUCCESS] Diag Watchdog Test \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_WATCHDOG_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the VIM RAM ECC diagnostic.
 *
 *  @param[in] cfg
 *      Configuration to be used in order to execute the diagnostic test.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_vimEccTest (Diag_VIM_ECC_Cfg cfg, uint32_t* ptrCycleCount)
{
    int32_t                 retVal;
    CSL_VIMRegs *ptrVIMRegs = CSL_VIM_getBaseAddress ();
    /* backup the FallBackVector address */
    uint32_t fallbackAddOrig = CSL_VIM_getFallbackVectorAddress(ptrVIMRegs);

    CSL_VIM_setFallbackVectorAddress(ptrVIMRegs,(uint32_t) SOC_MSS_ESM_LOW_PRIORITY_INT);
    /* Execute the diagnostic: */
    retVal = Diag_VIM_ECC_execute (&cfg, ptrCycleCount);
    /* restore the fallback address to VIM RAM */
    CSL_VIM_setFallbackVectorAddress(ptrVIMRegs,(uint32_t)fallbackAddOrig);
    if(retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM ECC %d-bit Error \n", cfg.eccMode, retVal);
        goto EXIT;
    }
    MMW_PRINT("[SUCCESS] Diag VIM ECC %d-bit Error \n", cfg.eccMode);
EXIT:

    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (\
                DIAG_MSS_VIM_ECC_1B_TEST_STATUS_BIT+(cfg.eccMode-1)));
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the ESM Static configuration
 *      diagnostic Test, to init, compare & check.
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.

 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_vimStaticTest(Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    uint32_t                    cycleCount = 0;

    /* Initialization VIM Static CFG DIAG */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Configuration [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Compare Static Config: DIAG */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Configuration [Compare] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag VIM StaticCfg Configuration \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_VIM_STATIC_TEST_STATUS_BIT);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the VIM static configuration diagnostic once the steady
 *      state has been modified. The test will modify the VIM 
 *      wakeup interrupt configuration which changes the steady state.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.

 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_vimStaticCfg_verifyViolationWakeup (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    CSL_VIMRegs*                ptrVIMRegs;
    uint8_t                     status;
    uint32_t                    cycleCount = 0;

    /* Get the VIM registers: */
    ptrVIMRegs = CSL_VIM_getBaseAddress ();

    /* Initialize VIM Static CFG: Diag */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation Wakeup [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Violate the steady state by modifying the interrupt wakeup register.
     * read the current status of wake up interrupt for channel 126 and flip it */
    status = CSL_VIM_getWakeupInterruptStatus(ptrVIMRegs, 126);
    if (status == 1U)
    {
        CSL_VIM_disableWakeupInterrupt(ptrVIMRegs, 126);
    }
    else
    {
        CSL_VIM_enableWakeupInterrupt(ptrVIMRegs, 126);
    }

    /* Execute the diagnostic: Comapre static CFG - This should fail. */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation Wakeup [Post steady state change] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The wakeup interrupt register was modified and that is the offset that should be returned */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_VIMRegs, WAKEENASET[3]))
    {
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation Wakeup [Offset mismatch]");
        goto EXIT;
    }

    /* Reset the diagnostic - The wakeup configuration is now in the baseline. */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation Wakeup [Reset] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Update the steady state by restoring the interrupt wakeup register to 
     * original value (it was flipped once and we are flipping it back)
     ********************************************************************************/
    /* read the current status of wake up interrupt for channel 126 and flip it */
    status = CSL_VIM_getWakeupInterruptStatus(ptrVIMRegs, 126);
    if (status == 1U)
    {
        CSL_VIM_disableWakeupInterrupt(ptrVIMRegs, 126);
    }
    else
    {
        CSL_VIM_enableWakeupInterrupt(ptrVIMRegs, 126);
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag VIM StaticCfg Verify Violation Wakeup \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_VIM_STATIC_TEST_STATUS_BIT);

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the VIM static configuration diagnostic once the steady
 *      state has been modified. The test will modify the VIM 
 *      Fallback vector address which changes the steady state.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.

 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_vimStaticCfg_verifyViolationFallbackAddr (Diag_StaticCfgErrInfo* ptrErrorInfo,
                                                          uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    CSL_VIMRegs*                ptrVIMRegs;
    uint32_t                    fallbackAddr;
    uint32_t                    cycleCount = 0;

    /* Get the VIM registers: */
    ptrVIMRegs = CSL_VIM_getBaseAddress ();

    /* Initialize VIM Static CFG : Diag */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation FallbackAddr [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Violate the steady state by modifying the Fallback vector address register
     ********************************************************************************/
    /* store the current fallback vector address */
    fallbackAddr = CSL_VIM_getFallbackVectorAddress (ptrVIMRegs);
    CSL_VIM_setFallbackVectorAddress(ptrVIMRegs, 0xdeadbeef);

    /* Execute the diagnostic: Comapre Static Cfg - This should fail. */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation FallbackAddr [Post steady state change] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The fallback vector addr register was modified and that is the offset that should be returned */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_VIMRegs, FBVECADDR))
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation FallbackAddr [Offset mismatch] \n", retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Reset the diagnostic:
     * - The fallback address configuration is now in the baseline.
     ********************************************************************************/
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT  ("[ERROR] Diag VIM StaticCfg Verify Violation FallbackAddr [Reset] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Restore the Fallback vector address register to the saved value
     ********************************************************************************/
    CSL_VIM_setFallbackVectorAddress(ptrVIMRegs, fallbackAddr);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag VIM StaticCfg Verify Violation FallbackAddr \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_VIM_STATIC_TEST_STATUS_BIT);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the VIM static configuration diagnostic once the steady
 *      state has been modified. The test will modify the VIM 
 *      TEST_DIAG_EN bits in ECCCTL which changes the steady state.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.

 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_vimStaticCfg_verifyViolationECCDiag (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    CSL_VIMRegs*                ptrVIMRegs;
    uint32_t                    cycleCount = 0;

    /* Get the VIM registers: */
    ptrVIMRegs = CSL_VIM_getBaseAddress ();

    /* Initialize VIM Static CFG : Diag */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ECCDiag [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Violate the steady state by modifying the TEST_DIAG_EN bits in ECCCTL register
     ********************************************************************************/
    /* the test_diag_en bits would be off by default and would only be enabled/disabled by a diagnostic */
    CSL_VIM_enableECCDiagnostic(ptrVIMRegs);

    /********************************************************************************
     * Execute the diagnostic: Compare VIM Static CFg
     * - This should fail.
     ********************************************************************************/
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ECCDiag [Post steady state change] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The wakeup interrupt register was modified and that is the offset that should be returned */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_VIMRegs, ECCCTL))
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ECCDiag [Offset mismatch] \n", retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Disable the TEST_DIAG_EN bits in ECCCTL register
     ********************************************************************************/
    CSL_VIM_disableECCDiagnostic(ptrVIMRegs);

    /********************************************************************************
     * Reset the diagnostic static cfg is now restored and in the baseline.
     ********************************************************************************/
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Failure Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ECCDiag [Reset] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT("[SUCCESS] Diag VIM StaticCfg Verify Violation ECCDiag \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_VIM_STATIC_TEST_STATUS_BIT);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for the
 *      VIM static configuration diagnostic once the steady state
 *      has been modified. The test will modify the VIM Interrupt
 *      Control Register (CHANCTRL) which changes the steady state.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.

 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_vimStaticCfg_verifyViolationChanCtrl (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    CSL_VIMRegs*                ptrVIMRegs;
    uint8_t                     intNum;
    uint8_t                     channelNum;
    uint32_t                    cycleCount = 0U;

    /* Get the VIM registers: */
    ptrVIMRegs = CSL_VIM_getBaseAddress ();

    /* Initialize VIM Static Cfg : DIag */
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ChanCtrl [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    channelNum = 120U;
    /* Store the interrupt associated with selected channelNum */
    intNum = CSL_VIM_getInterruptForChannel(ptrVIMRegs, channelNum);

    /********************************************************************************
     * Violate the steady state by modifying the CHANCTRL register
     ********************************************************************************/
    /* Set interrupt number to one more than the current intNum */
    CSL_VIM_setInterruptForChannel(ptrVIMRegs, (intNum + 1), channelNum);

    /********************************************************************************
     * Execute the diagnostic: compare static CFG
     * - This should fail.
     ********************************************************************************/
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ChanCtrl [Post steady state change] \n",\
                retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The CHANCTRL[30] register was modified and that is the offset that should be returned */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_VIMRegs, CHANCTRL[30]))
    {
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ChanCtrl [Offset mismatch] \n");
        goto EXIT;
    }

    /********************************************************************************
     * Restore the interrupt number for the channel
     ********************************************************************************/
    CSL_VIM_setInterruptForChannel(ptrVIMRegs, intNum, channelNum);

    /********************************************************************************
     * Reset the diagnostic static cfg:
     * - The CHANCTRL[30] is now restored and in the baseline.
     ********************************************************************************/
    retVal = Diag_VIM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag VIM StaticCfg Verify Violation ChanCtrl [Reset] [%d] \n",\
                retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag VIM StaticCfg Verify Violation ChanCtrl \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_VIM_STATIC_TEST_STATUS_BIT);

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      This is the implementation of the DMA MPU diagnostic.
 *
 *  @param[in] cfg
 *      Configuration to be used to execute the diagnostic
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_dmaMpuTest (Diag_DMA_MPU_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_RCMRegs*    ptrRCMRegs;
    CSL_DMARegs*    ptrDMARegs;
    uint8_t         done;
    int32_t             retVal;
    uint8_t             dmaChannel;
    uint8_t             region;
    uint32_t            memoryRegionCfg[4];
    uint8_t             memoryRegionIndex;
    uint32_t            cycleCount = 0;
    CSL_RCM_MemoryType  dmaMemType;
    
    /******************************************************************
     * DMA Module Initialization:
     ******************************************************************/
    /* Get the RCM Module */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    if(cfg.dmaInstanceId == 0)
    {
        dmaMemType = CSL_RCM_MemoryType_DMA;
    }
    else
    {
        dmaMemType = CSL_RCM_MemoryType_DMA2;
    }

    /* Initialize the memory for the DMA Instance requested */
    CSL_RCM_initMemory (ptrRCMRegs, dmaMemType);

    /* Initialize the status: */
    done = 0U;

    /* Wait for the memory initialization to be done: */
    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_RCM_isInitMemoryComplete (ptrRCMRegs, dmaMemType) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* initialize the DMA module: */
    /* Get the base address of the DMA Module: */
    ptrDMARegs = CSL_DMA_getBaseAddress (cfg.dmaInstanceId);

    /* SW Reset: */
    CSL_DMA_swReset (ptrDMARegs);

    /* Enable the DMA Instance: */
    CSL_DMA_setEnableStatus (ptrDMARegs, 1U);

    /* For each DMA Instance: Ensure that all the channels are mapped to
     * the valid Port as specified in the TRM. */
    for (dmaChannel = DMA_ERR_INJECTION_START_CHAN_IDX; dmaChannel <= \
         DMA_ERR_INJECTION_END_CHAN_IDX; dmaChannel++)
    {
        /* Set the DMA Channel To Port Mapping: */
        CSL_DMA_setDMAChannelToPortAssign (ptrDMARegs, dmaChannel, 0x7U);
    }

    /* Create the memory region configuration:
     * - Use TCMA, TCMB, L3 and HSRAM */
    memoryRegionCfg[0] = (uint32_t)&gTCMAMemoryRegion;
    memoryRegionCfg[1] = (uint32_t)&gTCMBMemoryRegion;
    memoryRegionCfg[2] = (uint32_t)&gL3MemoryRegion;
    memoryRegionCfg[3] = (uint32_t)&gHSRAMMemoryRegion;

    /* For selected DMA Channels: */
    for (dmaChannel = DMA_ERR_INJECTION_START_CHAN_IDX; dmaChannel <=\
        DMA_ERR_INJECTION_END_CHAN_IDX; dmaChannel++)
    {
        /* For all MPU Regions: */
        for (region = DMA_ERR_INJECTION_START_MPU_IDX; region <=\
             DMA_ERR_INJECTION_END_MPU_IDX; region++)
        {
            /* For all possible MPU Region Start Address: */
            for (memoryRegionIndex = 0U; memoryRegionIndex < 4U; memoryRegionIndex++)
            {
                /* Populate the diagnostic configuration: */
                cfg.dmaChannel         = dmaChannel;
                cfg.mpuRegion          = region;
                cfg.regionStartAddress = (uint32_t)memoryRegionCfg[memoryRegionIndex];

                /* Execute the diagnostic: */
                retVal = Diag_DMA_MPU_execute (&cfg, ptrCycleCount);
                if (retVal != DIAG_SUCCESS)
                {
                    MMW_PRINT ("[ERROR] Diag DMA MPU Failed Instance[%d] RegionAddr [0x%x] Channel[%d] Region[%d] [%d]\n",
                               cfg.dmaInstanceId, cfg.regionStartAddress, dmaChannel, region, retVal);
                    goto EXIT;
                }
                /* update the cyclecount if requested by the func caller */
                UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

            }
        }
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT("[SUCCESS] Diag DMA-%d MPU \n", cfg.dmaInstanceId);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_DMA0_MPU_TEST_STATUS_BIT+cfg.dmaInstanceId));

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the DMA Parity diagnostic.
 *
 *  @param[in] cfg
 *      Pointer to the configuration to be used to execute the diagnostic
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_dmaParityTest (Diag_DMA_Parity_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_RCMRegs*    ptrRCMRegs;
    CSL_DMARegs*    ptrDMARegs;
    uint8_t         done;
    uint8_t         dmaChannel;
    int32_t             retVal;
    uint32_t        cycleCount = 0;
    CSL_RCM_MemoryType  dmaMemType;

    /******************************************************************
     * DMA Module Initialization:
     ******************************************************************/
    /* Get the RCM Module */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    if(cfg.dmaInstanceId == 0)
    {
        dmaMemType = CSL_RCM_MemoryType_DMA;
    }
    else
    {
        dmaMemType = CSL_RCM_MemoryType_DMA2;
    }

    /* Initialize the memory for both the DMA Instance1: */
    CSL_RCM_initMemory (ptrRCMRegs, dmaMemType);

    /* Initialize the status: */
    done = 0U;

    /* Wait for the memory initialization to be done: */
    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_RCM_isInitMemoryComplete (ptrRCMRegs, dmaMemType) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Get the base address of the DMA Module: */
    ptrDMARegs = CSL_DMA_getBaseAddress (dmaMemType);

    /* SW Reset: */
    CSL_DMA_swReset (ptrDMARegs);

    /* Enable the DMA Instance: */
    CSL_DMA_setEnableStatus (ptrDMARegs, 1U);

    /* For each DMA Instance: Ensure that all the channels are mapped to
     * the valid Port as specified in the TRM. */
    for (dmaChannel = 0U; dmaChannel < 32U; dmaChannel++)
    {
        /* Set the DMA Channel To Port Mapping: */
        CSL_DMA_setDMAChannelToPortAssign (ptrDMARegs, dmaChannel, 0x7U);
    }

    /* For all DMA Channels: */
    for (dmaChannel = DMA_ERR_INJECTION_START_CHAN_IDX; dmaChannel <= \
        DMA_ERR_INJECTION_END_CHAN_IDX; dmaChannel++)
    {
        /* Populate the diagnostic configuration: */
        cfg.dmaChannel      = dmaChannel;

        /* Execute the diagnostic: */
        retVal = Diag_DMA_Parity_execute (&cfg, ptrCycleCount);
        if (retVal != DIAG_SUCCESS)
        {
            MMW_PRINT ("[ERROR] Diag DMA Parity Failed Instance[%d] Channel[%d] [%d]\n",
                       cfg.dmaInstanceId, dmaChannel, retVal);
            goto EXIT;
        }
        /* update the cyclecount if requested by the func caller */
        UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT("[SUCCESS] Diag DMA-%d Parity \n", cfg.dmaInstanceId);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_DMA0_PARITY_TEST_STATUS_BIT+cfg.dmaInstanceId));

    return 0;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the DMA Static configuration
 *      diagnostic test for both of DMA instance IDs.
 *
 *  @param[in] dmaInstanceId : DMA instance ID
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_dmaStaticTest (uint8_t dmaInstanceId, Diag_StaticCfgErrInfo *ptrErrorInfo,
                               uint32_t* ptrCycleCount)
{
    CSL_DMARegs*    ptrDMARegs;
    CSL_RCMRegs*    ptrRCMRegs;
    CSL_DMA_RAMRegs*            ptrDMARAMRegs;
    int32_t                     retVal;
    Diag_DMA_StaticCfgParams    params;
    uint8_t  done;
    uint32_t                    srcData;
    uint32_t                    dstData;
    uint8_t                     dmaChannel;
    uint32_t                    cycleCount = 0;
    CSL_RCM_MemoryType          dmaMemType;
    /******************************************************************
     * DMA Module Initialization:
     ******************************************************************/
    /* Get the RCM Module */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    if(dmaInstanceId == 0)
    {
        dmaMemType = CSL_RCM_MemoryType_DMA;
    }
    else
    {
        dmaMemType = CSL_RCM_MemoryType_DMA2;
    }

    /* Initialize the memory for DMA Instance */
    CSL_RCM_initMemory (ptrRCMRegs, dmaMemType);

    /* Initialize the status: */
    done = 0U;

    /* Wait for the memory initialization to be done: */
    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_RCM_isInitMemoryComplete (ptrRCMRegs, dmaMemType) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }
    /* Get the base address of the DMA Module: */
    ptrDMARegs = CSL_DMA_getBaseAddress (dmaInstanceId);

    /* SW Reset: */
    CSL_DMA_swReset (ptrDMARegs);

    /* Enable the DMA Instance: */
    CSL_DMA_setEnableStatus (ptrDMARegs, 1U);

    /* Cycle through all the DMA Channels: */
    for (dmaChannel = DMA_ERR_INJECTION_START_CHAN_IDX; dmaChannel <=\
        DMA_ERR_INJECTION_START_CHAN_IDX; dmaChannel++)
    {
        /* Set the DMA Channel To Port Mapping: */
        CSL_DMA_setDMAChannelToPortAssign (ptrDMARegs, dmaChannel, 0x7U);

        /********************************************************************************
         * Initialize the diagnostic:
         ********************************************************************************/
        (void)memset ((void *)&params, 0, sizeof(Diag_DMA_StaticCfgParams));

        /* Populate the parameters: */
        params.mode          = Diag_StaticCfgMode_INIT;
        params.dmaInstanceId = dmaInstanceId;

        /* Execute the diagnostic: */
        retVal = Diag_DMA_staticCfgExecute (&params, ptrErrorInfo, ptrCycleCount);
        if (retVal != DIAG_SUCCESS)
        {
            /*  Test Result Reporting: */
            MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify DMA [Initialization] \n", retVal);
            goto EXIT;
        }
        /* update the cyclecount if requested by the func caller */
        UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

        /********************************************************************************
         * Initiate the DMA:
         ********************************************************************************/

        /* Get the Base Address of the DMA: */
        ptrDMARegs    = CSL_DMA_getBaseAddress(dmaInstanceId);
        ptrDMARAMRegs = CSL_DMA_getRAMBaseAddress(dmaInstanceId);

        /* Initialize the contents of the source & destination */
        srcData = 0xdeaddead;
        dstData = 0x0;

        /* Source & Destination Address: */
        CSL_DMA_setInitialSrcAddress (ptrDMARAMRegs, dmaChannel, (uint32_t)&srcData);
        CSL_DMA_setInitialDstAddress (ptrDMARAMRegs, dmaChannel, (uint32_t)&dstData);

        /* Single frame with buffer size elements: */
        CSL_DMA_setInitialTransferCount (ptrDMARAMRegs, dmaChannel, (uint16_t)1U, (uint16_t)sizeof(dstData));

        /* No chaining: */
        CSL_DMA_setChainChannel (ptrDMARAMRegs, dmaChannel, 0U);

        /* Read & Write element size is 8bits */
        CSL_DMA_setReadElementSize (ptrDMARAMRegs,  dmaChannel, 0U);
        CSL_DMA_setWriteElementSize (ptrDMARAMRegs, dmaChannel, 0U);

        /* Transfer is a single frame: */
        CSL_DMA_setTransferType (ptrDMARAMRegs, dmaChannel, 0U);

        /* Indexed Read/Write transfers: */
        CSL_DMA_setReadMode (ptrDMARAMRegs,  dmaChannel, 1U);
        CSL_DMA_setWriteMode (ptrDMARAMRegs, dmaChannel, 1U);

        /* No element/frame offsets: */
        CSL_DMA_setElementIndexOffset (ptrDMARAMRegs, dmaChannel, 0U, 0U);
        CSL_DMA_setFrameIndexOffset   (ptrDMARAMRegs, dmaChannel, 0U, 0U);

        /* Auto-Initiation mode is disabled. */
        CSL_DMA_setAutoInitiationMode (ptrDMARAMRegs, dmaChannel, 0U);

        /*************************************************************
         * Trigger the DMA:
         *************************************************************/
        CSL_DMA_setSWTrigger (ptrDMARegs, dmaChannel, 1U);

        /* Wait for the DMA Transaction to be completed: */
        while (CSL_DMA_isFTCPending (ptrDMARegs, dmaChannel) == 0);

        /* Sanity Check: Validate the DMA Transfer completed. */
        if (srcData != dstData)
        {
            /*  Test Result Reporting: */
            MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify DMA [DMA Completion] \n", retVal);
            goto EXIT;
        }

        /********************************************************************************
         * Execute the diagnostic:
         ********************************************************************************/
        (void)memset ((void *)&params, 0, sizeof(Diag_DMA_StaticCfgParams));

        /* Populate the parameters: */
        params.mode          = Diag_StaticCfgMode_COMPARE;
        params.dmaInstanceId = dmaInstanceId;

        /* Execute the diagnostic: */
        retVal = Diag_DMA_staticCfgExecute (&params, ptrErrorInfo, ptrCycleCount);
        if (retVal != DIAG_SUCCESS)
        {
            MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify DMA [Post DMA Completion] \n", retVal);
            goto EXIT;
        }
        /* update the cyclecount if requested by the func caller */
        UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

        /* Clear the FTC Interrupt status: */
        CSL_DMA_clearFTC (ptrDMARegs, dmaChannel);
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag DMA StaticCfg Configuration \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_DMA0_STATIC_TEST_STATUS_BIT+dmaInstanceId));

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the DMA static configuration diagnostic once the steady
 *      state has been modified. The test will modify the MPU
 *      configuration which changes the steady state.
 *
 *  @param[in] dmaInstanceId : DMA instance ID
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_dmaStaticCfg_verifyMPU (uint8_t dmaInstanceId, Diag_StaticCfgErrInfo* ptrErrorInfo,
                                        uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    Diag_DMA_StaticCfgParams    params;
    CSL_DMARegionCfg            regionCfg;
    CSL_DMARegs*                ptrDMARegs;
    uint32_t                    startAddress;
    uint32_t                    endAddress;
    uint32_t                    cycleCount = 0;

    /* Get the DMA registers: */
    ptrDMARegs = CSL_DMA_getBaseAddress (dmaInstanceId);

    /********************************************************************************
     * Initialize the diagnostic:
     ********************************************************************************/
    (void)memset ((void *)&params, 0, sizeof(Diag_DMA_StaticCfgParams));

    /* Populate the parameters: */
    params.mode          = Diag_StaticCfgMode_INIT;
    params.dmaInstanceId = dmaInstanceId;

    /* Execute the diagnostic: */
    retVal = Diag_DMA_staticCfgExecute (&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify MPU [Initialization] \n", \
                retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Update the steady state by modifying the MPU configuration
     ********************************************************************************/
    regionCfg.startAddress      = (uint32_t)&startAddress;
    regionCfg.endAddress        = (uint32_t)&endAddress;
    regionCfg.interruptStatus   = 1U;
    regionCfg.permissions       = CSL_DMA_Permissions_WRITE_ONLY;
    regionCfg.enable            = 1U;

    /* Set the Region configuration */
    CSL_DMA_setRegionCfg(ptrDMARegs, 0U, &regionCfg);

    /********************************************************************************
     * Execute the diagnostic:
     * - This should fail.
     ********************************************************************************/
    (void)memset ((void *)&params, 0, sizeof(Diag_DMA_StaticCfgParams));

    /* Populate the parameters: */
    params.mode          = Diag_StaticCfgMode_COMPARE;
    params.dmaInstanceId = dmaInstanceId;

    /* Execute the diagnostic: */
    retVal = Diag_DMA_staticCfgExecute (&params, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify MPU [Post MPU] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The DMA Protection Control register would have been modified
     *   since the MPU has been changed */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_DMARegs, DMAMPCTRL))
    {
        MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify MPU [Post MPU] \n", retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Reset the diagnostic:
     * - The MPU configuration is now in the baseline.
     ********************************************************************************/
    (void)memset ((void *)&params, 0, sizeof(Diag_DMA_StaticCfgParams));

    /* Populate the parameters: */
    params.mode          = Diag_StaticCfgMode_RESET;
    params.dmaInstanceId = dmaInstanceId;

    /* Execute the diagnostic: */
    retVal = Diag_DMA_staticCfgExecute (&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag DMA Static Configuration Verify MPU [Reset] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag DMA Static Configuration Verify MPU \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_DMA0_STATIC_TEST_STATUS_BIT+dmaInstanceId));

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the MIBSPI ECC diagnostic.
 *      It injects single or double bit errors on the Rx or Tx RAM
 *      and verifies the ECC protection.
 *
 *  @param[in] cfg
 *      Configuration used to execute the diagnostic.
 *  @param[out] ptrErrorInfo
 *      Pointer to the error information.
 *      This field is populated ONLY when there is an error detected by the diagnostic.
 *      \ref DIAG_EDATA
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_MibspiEccTest (Diag_MIBSPI_ECC_Cfg  cfg,
                               Diag_MIBSPI_ECC_ErrorInfo* ptrErrorInfo,
                               uint32_t* ptrCycleCount)
{
    CSL_MIBSPIRegs*     ptrMIBSPIRegs;
    uint8_t             done;
    uint32_t                    cycleCount = 0;
    uint8_t                     ramBufferIndex;
    int32_t                     retVal;
    
    /******************************************************************
     * MIBSPI Module Initialization:
     ******************************************************************/
    /* Get the base address of the MIBSPI Module */
    ptrMIBSPIRegs = CSL_MIBSPI_getBaseAddress(cfg.instanceId);

    /* Reset the module */
    CSL_MIBSPI_reset(ptrMIBSPIRegs);

    /* Wait for the Multibuffer RAM initialization to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_MIBSPI_getMultiBufferInitStatus(ptrMIBSPIRegs) == 0U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Enable MIBSPI multi buffer mode for both instances. */
    /* Get the base address of the MIBSPI Module */
    ptrMIBSPIRegs = CSL_MIBSPI_getBaseAddress(cfg.instanceId);

    /* Enable multi buffer mode */
    CSL_MIBSPI_setMultiBufferModeStatus(ptrMIBSPIRegs, 1U);

    /***************************************************************************
     * Diagnostic: TCMA Parity
     ***************************************************************************/

    /* Execute for all instances. */
    for (ramBufferIndex = MIBSPI_RAM_ECC_START_IDX; ramBufferIndex <= MIBSPI_RAM_ECC_END_IDX; ramBufferIndex++)
    {
        cfg.ramBufferIndex = ramBufferIndex;
        /* Populate the configuration for TX RAM Type */
        cfg.ramBufferType   = CSL_MIBSPI_RamType_TX;

        /* Execute the diagnostic. */
        retVal = Diag_MIBSPI_ECC_execute(&cfg, ptrErrorInfo, ptrCycleCount);
        if (retVal != DIAG_SUCCESS)
        {
            MMW_PRINT("[ERROR] MIBSPI ECC Diagnostic [SB TX RAM Error Instance[%d] RAM Buffer Index[%d]] [%d]\n", \
                      cfg.instanceId, ramBufferIndex, retVal);

            /* Test failed. */
            goto EXIT;
        }
        /* update the cyclecount if requested by the func caller */
        UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);
        MMW_PRINT("[SUCCESS] MIBSPI-%d ECC Diagnostic TX RAM, Buffer Index[%d] \n", \
                  cfg.instanceId, ramBufferIndex);
        /* Populate the configuration for RX RAM Type*/
        cfg.ramBufferType   = CSL_MIBSPI_RamType_RX;

        /* Execute the diagnostic. */
        retVal = Diag_MIBSPI_ECC_execute(&cfg, ptrErrorInfo, ptrCycleCount);
        if (retVal != DIAG_SUCCESS)
        {
            MMW_PRINT("[ERROR] MIBSPI ECC Diagnostic [SB RX RAM Error Instance[%d] RAM Buffer Index[%d]] [%d]\n", \
                      cfg.instanceId, ramBufferIndex, retVal);

            /* Test failed. */
            goto EXIT;
        }
        /* update the cyclecount if requested by the func caller */
        UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);
        MMW_PRINT("[SUCCESS] MIBSPI-%d ECC %d-Bit Diagnostic RX RAM, Buffer Index[%d] \n", \
                  cfg.instanceId, cfg.eccMode, ramBufferIndex);
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
EXIT:
    if ((cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR) && (cfg.instanceId == 0))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MIBSPI0_ECC_1B_TEST_STATUS_BIT);
    }
    else if ((cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR) && (cfg.instanceId == 1))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MIBSPI1_ECC_1B_TEST_STATUS_BIT);
    }
    else if ((cfg.eccMode == Diag_ECCMode_DOUBLE_BIT_ERROR) && (cfg.instanceId == 0))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MIBSPI0_ECC_2B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MIBSPI1_ECC_2B_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the MIBSPI Static configuration
 *      diagnostic.
 *
 *  @param[in] instanceId       MibSPI Instance ID
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_MibspiStaticTest (uint8_t instanceId,
                                  Diag_StaticCfgErrInfo* ptrErrorInfo,
                                  uint32_t* ptrCycleCount)
{
    CSL_MIBSPIRegs*     ptrMIBSPIRegs;
    uint8_t             done;
    uint32_t            cycleCount = 0;
    Diag_MIBSPI_StaticCfgParams     params;
    int32_t                         retVal;

    /******************************************************************
     * MIBSPI Module Initialization:
     ******************************************************************/
    /* Get the base address of the MIBSPI Module */
    ptrMIBSPIRegs = CSL_MIBSPI_getBaseAddress(instanceId);

    /* Reset the module */
    CSL_MIBSPI_reset(ptrMIBSPIRegs);

    /* Wait for the Multibuffer RAM initialization to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_MIBSPI_getMultiBufferInitStatus(ptrMIBSPIRegs) == 0U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Enable multi buffer mode */
    CSL_MIBSPI_setMultiBufferModeStatus(ptrMIBSPIRegs, 1U);

    (void)memset((void *)&params, 0, sizeof(Diag_MIBSPI_StaticCfgParams));
    /* Populate the parameters. */
    params.instanceId   = instanceId;
    params.mode         = Diag_StaticCfgMode_INIT;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_staticCfgExecute(&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag MIBSPI [%d] Static Configuration [MIBSPIA Init][%d]\n", \
               instanceId, retVal);
        /* Test failed. */
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
    * Compare Cycle count for MIBSPIA/B
    ********************************************************************/
    (void)memset((void *)&params, 0, sizeof(Diag_MIBSPI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = 0U;
    params.mode         = Diag_StaticCfgMode_COMPARE;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_staticCfgExecute(&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag MIBSPI [%d] Static Configuration [MIBSPIA Compare] Instance[%d]\n", \
               instanceId, retVal);
        /* Test failed. */
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
    *  Reset Cycle count for MIBSPIA/B
    ********************************************************************/
    (void)memset((void *)&params, 0, sizeof(Diag_MIBSPI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = 0U;
    params.mode         = Diag_StaticCfgMode_RESET;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_staticCfgExecute(&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag MIBSPI [%d] Static Configuration [MIBSPIA Reset] Instance[%d]\n", \
               instanceId, retVal);
        /* Test failed. */
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag MibSPI-%d StaticCfg Configuration \n", instanceId);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_MIBSPI0_STATIC_TEST_STATUS_BIT+instanceId));

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the MIBSPI static configuration diagnostic once the steady
 *      state has been modified. The test will modify the counter
 *      configuration which changes the steady state.
 *
 *  @param[in] instanceId       MibSPI Instance ID
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_MibSPiStaticViolationTest (uint8_t instanceId,
                                          Diag_StaticCfgErrInfo* ptrErrorInfo,
                                          uint32_t* ptrCycleCount)
{
    CSL_MIBSPIRegs*     ptrMIBSPIRegs;
    uint8_t             done;
    uint32_t            cycleCount = 0;
    Diag_MIBSPI_StaticCfgParams     params;
    int32_t                         retVal;

    /******************************************************************
     * MIBSPI Module Initialization:
     ******************************************************************/
    /* Get the base address of the MIBSPI Module */
    ptrMIBSPIRegs = CSL_MIBSPI_getBaseAddress(instanceId);

    /* Reset the module */
    CSL_MIBSPI_reset(ptrMIBSPIRegs);

    /* Wait for the Multibuffer RAM initialization to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_MIBSPI_getMultiBufferInitStatus(ptrMIBSPIRegs) == 0U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Enable multi buffer mode */
    CSL_MIBSPI_setMultiBufferModeStatus(ptrMIBSPIRegs, 1U);

    /********************************************************************************
     * Initialize the diagnostic:
     ********************************************************************************/
    (void)memset((void *)&params, 0, sizeof(Diag_MIBSPI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = instanceId;
    params.mode         = Diag_StaticCfgMode_INIT;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_staticCfgExecute(&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag MIBSPI [%d] Static Configuration Verify Config [Initialization] \n", \
               instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Update the steady state by modifying the configuration
     ********************************************************************************/
    /* Get the base address of the MIBSPI instance. */
    ptrMIBSPIRegs = CSL_MIBSPI_getBaseAddress(instanceId);

    /* Disable multibuffer mode */
    CSL_MIBSPI_setMultiBufferModeStatus(ptrMIBSPIRegs, 0U);

    /********************************************************************************
     * Execute the diagnostic:- This should fail.
    ********************************************************************************/
    (void)memset((void *)&params, 0, sizeof(Diag_MIBSPI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = instanceId;
    params.mode         = Diag_StaticCfgMode_COMPARE;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_staticCfgExecute(&params, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag MIBSPI [%d] Static Configuration Verify Config [Steady state change] \n",\
               instanceId);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The MIBSPI enable register would have been modified
     */
    if (ptrErrorInfo->errorOffset != offsetof(CSL_MIBSPIRegs, MIBSPIE))
    {
        MMW_PRINT("[ERROR] Diag MIBSPI [%d] Static Configuration Verify Config [Steady state change - Offset mismatch] \n",\
               instanceId, retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Reset the diagnostic:
     * - The counter configuration is now in the baseline.
     ********************************************************************************/
    (void)memset((void *)&params, 0, sizeof(Diag_MIBSPI_StaticCfgParams));

    /* Populate the parameters. */
    params.instanceId   = instanceId;
    params.mode         = Diag_StaticCfgMode_RESET;

    /* Execute the diagnostic. */
    retVal = Diag_MIBSPI_staticCfgExecute(&params, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag MIBSPI-%d Static Configuration Verify Config [Reset] \n", instanceId, retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT("[SUCCESS] Diag MIBSPI-%d Static Configuration Verify Config \n", instanceId);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_MIBSPI0_STATIC_TEST_STATUS_BIT+instanceId));

   return retVal;
}

#ifndef SOC_XWR68XX

/**
 *  @b Description
 *  @n
 *      The function is used to test the DCAN loopback Diagnostic
 *      with basic configuration. The results of the test execution
 *      and the time taken to execute the diagnostic are logged into
 *      cycleCount
 *
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *
 */
int32_t MssDiag_DcanLoopbackTest(uint32_t* ptrCycleCount)
{
    CSL_DCANRegs*               ptrDCANRegs;
    Diag_DCAN_LoopbackType      mode;
    uint32_t                    cycleCount = 0;
    int32_t                     retVal;
    /******************************************************************
     * DCAN Module Initialization:
     ******************************************************************/

    /* Get the base address of the DCAN module. */
    ptrDCANRegs = CSL_DCAN_getBaseAddress();

    /* Configure DCAN module in normal mode of operation. */
    CSL_DCAN_setInitStatus(ptrDCANRegs, 0U);

    /***************************************************************************
     * Diagnostic: DCAN Loopback Mode Disabled
     ***************************************************************************/
    /* Populate the configuration. */
    mode = Diag_DCAN_LoopbackType_DISABLED;

    /* Execute the diagnostic. */
    retVal = Diag_DCAN_loopbackExecute(mode, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag DCAN loopback [Mode Disabled] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: DCAN Loopback Mode Internal
     ***************************************************************************/
    /* Populate the configuration. */
    mode = Diag_DCAN_LoopbackType_INT;

    /* Execute the diagnostic. */
    retVal = Diag_DCAN_loopbackExecute(mode, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag DCAN loopback [Mode Internal] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: DCAN Loopback Mode External
     ***************************************************************************/
    /* Populate the configuration. */
    mode = Diag_DCAN_LoopbackType_EXT;

    /* Execute the diagnostic. */
    retVal = Diag_DCAN_loopbackExecute(mode, ptrCycleCount);
    if(retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag DCAN loopback [Mode External] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT ("[SUCCESS] Diag DCAN Loopback \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_DCAN_LOOPBACK_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the DCAN Static configuration
 *      diagnostic.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_DcanStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    CSL_DCANRegs*   ptrDCANRegs;
    CSL_RCMRegs*    ptrRCMRegs;
    uint8_t         done;
    uint32_t        cycleCount = 0;
    int32_t         retVal;
    
    /******************************************************************
     * DCAN Module Initialization:
     ******************************************************************/
    /* Get the RCM Module base address. */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    /* Initialize the DCAN memory. */
    CSL_RCM_initMemory (ptrRCMRegs, CSL_RCM_MemoryType_DCAN);

    /* Initialize the status. */
    done = 0U;

    /* Wait for the memory initialization to be done. */
    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_RCM_isInitMemoryComplete (ptrRCMRegs, CSL_RCM_MemoryType_DCAN) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Get the base address of the DCAN module. */
    ptrDCANRegs = CSL_DCAN_getBaseAddress();

    /* Configure DCAN module in normal mode of operation. */
    CSL_DCAN_setInitStatus(ptrDCANRegs, 0U);

    /********************************************************************
     * Initialization Cycle count for DCAN
     ********************************************************************/
 
    /* Execute the diagnostic. */
    retVal = Diag_DCAN_staticCfgExecute(Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag DCAN Static Configuration [DCAN Initialization] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Compare Cycle count for DCAN
     ********************************************************************/

    /* Execute the diagnostic. */
    retVal = Diag_DCAN_staticCfgExecute(Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag DCAN Static Configuration [DCAN Compare] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag DCAN StaticCfg Configuration \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_DCAN_STATIC_TEST_STATUS_BIT);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the DCAN static configuration diagnostic once the steady
 *      state has been modified. The test will modify the counter
 *      configuration which changes the steady state.
 *
 *  @retval
 *      Not applicable
 */
int32_t MssDiag_dcanStatic_verifyConfig (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    CSL_DCANRegs*          ptrDCANRegs;
    int32_t                retVal;
    uint8_t                done;
    CSL_RCMRegs*           ptrRCMRegs;
    uint32_t               cycleCount = 0;

    /* Get the RCM Module base address. */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    /* Initialize the DCAN memory. */
    CSL_RCM_initMemory (ptrRCMRegs, CSL_RCM_MemoryType_DCAN);

    /* Initialize the status. */
    done = 0U;

    /* Wait for the memory initialization to be done. */
    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_RCM_isInitMemoryComplete (ptrRCMRegs, CSL_RCM_MemoryType_DCAN) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Get the base address of the DCAN module. */
    ptrDCANRegs = CSL_DCAN_getBaseAddress();

    /* Configure DCAN module in normal mode of operation. */
    CSL_DCAN_setInitStatus(ptrDCANRegs, 0U);

    /********************************************************************************
     * Initialize the diagnostic:
     ********************************************************************************/
    retVal = Diag_DCAN_staticCfgExecute(Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag DCAN Static Configuration Verify Config [Initialization] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Update the steady state by modifying the configuration
     ********************************************************************************/

    /* Get the base address of the DCAN module. */
    ptrDCANRegs = CSL_DCAN_getBaseAddress();

    /* Put DCAN in power down mode of operation. */
    CSL_DCAN_setPowerDownStatus(ptrDCANRegs, 1U);

    /********************************************************************************
     * Execute the diagnostic:
     * - This should fail.
     ********************************************************************************/
    retVal = Diag_DCAN_staticCfgExecute(Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag DCAN Static Configuration Verify Config [Steady state change] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The DCAN enable register would have been modified
     */
    if (ptrErrorInfo->errorOffset != offsetof(CSL_DCANRegs, CTL))
    {
        MMW_PRINT("[ERROR] Diag DCAN Static Configuration Verify Config [Steady state change - Offset mismatch] \n", retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Reset the diagnostic:
     * - The counter configuration is now in the baseline.
     ********************************************************************************/

    /* Execute the diagnostic. */
    retVal = Diag_DCAN_staticCfgExecute(Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag DCAN Static Configuration Verify Config [Reset] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);
    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    MMW_PRINT("[SUCCESS] Diag DCAN Static Configuration Verify Config \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_DCAN_STATIC_TEST_STATUS_BIT);
    return retVal;
}
#endif


/**
 *  @b Description
 *  @n
 *      This is the implementation of the MCAN Static configuration
 *      diagnostic.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_McanStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    uint8_t             done;
    CSL_MCANRegs*       ptrMCANRegs;
    uint32_t            cycleCount = 0;
    int32_t             retVal;
    
    /******************************************************************
     * MCAN Module Initialization:
     ******************************************************************/
    /* Get the base address of the MCAN module. */
    ptrMCANRegs = CSL_MCAN_getBaseAddress();

    /* Reset the MCAN module. */
    CSL_MCAN_reset(ptrMCANRegs);

    /* Wait for the MCAN module reset to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the module still in reset? */
        if (CSL_MCAN_getResetStatus(ptrMCANRegs) == 0U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Configure MCAN module in initialization mode of operation.
     * We will be modifying configuration so leave the module in INIT mode.
    */
    CSL_MCAN_setInitStatus(ptrMCANRegs, 1U);

    /* Wait for the MCAN module initialization to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the module in Initialization mode? */
        if (CSL_MCAN_getInitStatus(ptrMCANRegs) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /********************************************************************
     * Initialization Cycle count for MCAN
     ********************************************************************/
    retVal = Diag_MCAN_staticCfgExecute(Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag MCAN Static Configuration [Initialization] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Compare Cycle count for MCAN
     ********************************************************************/
    retVal = Diag_MCAN_staticCfgExecute(Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag MCAN Static Configuration [Compare] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag MCAN StaticCfg Configuration \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MCAN_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the MCAN static configuration diagnostic once the steady
 *      state has been modified. The test will modify the loopback
 *      configuration which changes the steady state.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_mcanStatic_verifyConfig (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    CSL_MCANRegs*                   ptrMCANRegs;
    int32_t                         retVal;
    uint8_t                         loopback;
    uint32_t                        cycleCount = 0;
    uint8_t  done;
    
    /******************************************************************
     * MCAN Module Initialization:
     ******************************************************************/
    /* Get the base address of the MCAN module. */
    ptrMCANRegs = CSL_MCAN_getBaseAddress();

    /* Reset the MCAN module. */
    CSL_MCAN_reset(ptrMCANRegs);

    /* Wait for the MCAN module reset to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the module still in reset? */
        if (CSL_MCAN_getResetStatus(ptrMCANRegs) == 0U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Configure MCAN module in initialization mode of operation.
     * We will be modifying configuration so leave the module in INIT mode.
    */
    CSL_MCAN_setInitStatus(ptrMCANRegs, 1U);

    /* Wait for the MCAN module initialization to complete. */
    done = 0U;

    while (done == 0U)
    {
        /* Is the module in Initialization mode? */
        if (CSL_MCAN_getInitStatus(ptrMCANRegs) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /********************************************************************************
     * Initialize the diagnostic:
     ********************************************************************************/
    /* Get the base address of the MCAN module. */
    ptrMCANRegs = CSL_MCAN_getBaseAddress();

    /* Find the current MCAN loopback status */
    loopback = CSL_MCAN_getLoopbackStatus(ptrMCANRegs);

    /* Execute the diagnostic. */
    retVal = Diag_MCAN_staticCfgExecute(Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag MCAN Static Configuration Verify Config [Initialization] \n", \
               retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Update the steady state by modifying the configuration
     * Configure MCAN in loopback mode of operation
     ********************************************************************************/

    /* Enable CCE(configuration change enable) required to configure MCAN loopback. */
    CSL_MCAN_setConfigEnableStatus(ptrMCANRegs, 1U);
    /* Flip the loopback */
    loopback = loopback ^ 0x1U;
    CSL_MCAN_setLoopbackStatus(ptrMCANRegs, loopback);
    /* Disable CCE(configuration change enable) */
    CSL_MCAN_setConfigEnableStatus(ptrMCANRegs, 0U);

    /********************************************************************************
     * Execute the diagnostic:
     * - This should fail.
     ********************************************************************************/

    /* Execute the diagnostic. */
    retVal = Diag_MCAN_staticCfgExecute(Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag MCAN Static Configuration Verify Config [Steady state change] \n");
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The MCAN enable register would have been modified
     */
    if (ptrErrorInfo->errorOffset != offsetof(CSL_MCANRegs, MCAN_TEST))
    {
        MMW_PRINT("[ERROR] Diag MCAN Static Configuration Verify Config [Steady state change - Offset mismatch]\n");
        goto EXIT;
    }

    /********************************************************************************
     * Reset the diagnostic:
     * - The loopback configuration change is now in the baseline.
     ********************************************************************************/

    /* Execute the diagnostic. */
    retVal = Diag_MCAN_staticCfgExecute(Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag MCAN Static Configuration Verify Config [Reset] \n",\
               retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Update the steady state by modifying the configuration
     * Restore the loopback mode
     ********************************************************************************/

    /* Enable CCE(configuration change enable) required to configure MCAN loopback. */
    CSL_MCAN_setConfigEnableStatus(ptrMCANRegs, 1U);
    /* Restore the original loopback status */
    loopback = loopback ^ 0x1U;
    CSL_MCAN_setLoopbackStatus(ptrMCANRegs, loopback);
    /* Disable CCE(configuration change enable) */
    CSL_MCAN_setConfigEnableStatus(ptrMCANRegs, 0U);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT("[SUCCESS] Diag MCAN Static Configuration Verify Config \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_MCAN_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the Mailbox ECC diagnostic.
 *      It injects single or double bit errors in the specified mailbox
 *      and verifies the ECC protection.
 *
 *  @param[in] cfg
 *      configuration used to execute the diagnostic.
 *  @param[out] ptrErrorInfo
 *      Pointer to the error information.
 *      This field is populated ONLY when there is an error detected by the diagnostic.
 *      \ref DIAG_EDATA
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_MailboxEccTest (Diag_Mailbox_ECC_Cfg         cfg,
                                Diag_Mailbox_ECC_ErrorInfo* ptrErrorInfo,
                                uint32_t* ptrCycleCount)
{
    CSL_RCMRegs*   ptrRCMRegs;
    uint32_t       cycleCount = 0;
    int32_t        retVal;
    uint32_t       offset = 0U;
    uint8_t        done;
    uint8_t        errorAddr;

    /******************************************************************
     * Mailbox Memory Initialization:
     ******************************************************************/
    /* Get the RCM Module */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    /* Enable ECC */
    CSL_RCM_setMailboxECC(ptrRCMRegs, cfg.memType, 1U);

    /* Initialize the memory for specified mailbox */
    CSL_RCM_initMemory (ptrRCMRegs, cfg.memType);

    /* Initialize the status: */
    done = 0U;

    /* Wait for the memory initialization to be done: */
    while (done == 0U)
    {
        /* Is the memory initialization done? */
        if (CSL_RCM_isInitMemoryComplete (ptrRCMRegs, cfg.memType) == 1U)
        {
            /* YES: We can EXIT the loop. */
            done = 1U;
        }
    }

    /* Get the RCM Module */
    ptrRCMRegs = CSL_RCM_getBaseAddress();

    /***************************************************************************
     * Diagnostic: Mailbox ECC for mailboxMemoryType[mbox] eccMode[mode]
     ***************************************************************************/

    /* Execute the diagnostic. */
    retVal = Diag_Mailbox_ECC_execute(&cfg, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag Mailbox ECC [%s] \n", \
               mailboxMemoryTypeName[cfg.memType-CSL_RCM_MemoryType_MAILBOX_MSS_TO_BSS], retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    offset = offset + 16U; /* Offset has to be 8 byte aligned */

    /* Verify that the fault address is 0 after the diagnostic */
    errorAddr = CSL_RCM_getMailboxECCFaultAddress(ptrRCMRegs, cfg.memType);
    if (errorAddr != 0U)
    {
        MMW_PRINT("ECC fault address is non zero (0x%x) after MBOX test\n",
                       errorAddr);
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT("[SUCCESS] Diag Mailbox ECC %d-bit Error [%s] \n", cfg.eccMode, \
           mailboxMemoryTypeName[cfg.memType-CSL_RCM_MemoryType_MAILBOX_MSS_TO_BSS]);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (\
            DIAG_MSS_MAILBOX_ECC_1B_TEST_STATUS_BIT+(cfg.eccMode-1)));
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the TCM ECC diagnostic. It injects single
 *      or double bit errors on the program or data memory and verifies the ECC
 *      protection.

 *  @param[in] cfg
 *      Configuration used to execute the diagnostic for TCMA/TCMB0/TCMB1.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_TcmEccTest (Diag_TCM_ECC_Cfg cfg, uint32_t* ptrCycleCount)
{
    int32_t       retVal;

    /* Execute the diagnostic. */
    retVal = Diag_TCM_ECC_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag %s %d Bit ECC  \n",
                  tcmMemoryTypeName[cfg.memType-1], cfg.eccMode, retVal);
        goto EXIT;
    }
    MMW_PRINT("[SUCCESS] Diag %s %d Bit ECC  \n",
              tcmMemoryTypeName[cfg.memType-1], cfg.eccMode);

EXIT:
    if ((cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR) && (cfg.memType == CSL_RCM_MemoryType_TCMA))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_A_ECC_1B_TEST_STATUS_BIT);
    }
    else if ((cfg.eccMode == Diag_ECCMode_DOUBLE_BIT_ERROR) && (cfg.memType == CSL_RCM_MemoryType_TCMA))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_A_ECC_2B_TEST_STATUS_BIT);
    }
    else if ((cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR) && (cfg.memType == CSL_RCM_MemoryType_TCMB0))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_B0_ECC_1B_TEST_STATUS_BIT);
    }
    else if ((cfg.eccMode == Diag_ECCMode_DOUBLE_BIT_ERROR) && (cfg.memType == CSL_RCM_MemoryType_TCMB0))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_B0_ECC_2B_TEST_STATUS_BIT);
    }
    else if ((cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR) && (cfg.memType == CSL_RCM_MemoryType_TCMB1))
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_B1_ECC_1B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_B1_ECC_2B_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the TCM parity diagnostic. It forces a
 *      parity error on the program or data memory and verifies the parity
 *      protection.
 *
 *  @param[in] cfg
 *      Configuration used to execute the diagnostic for TCMA/TCMB0/TCMB1.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_TcmParityTest (Diag_TCM_Parity_Cfg cfg, uint32_t* ptrCycleCount)
{
    int32_t        retVal;

    /* Execute the diagnostic. */
    retVal = Diag_TCM_Parity_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting. */
        MMW_PRINT("[ERROR] Diag %s Parity [TCMA Parity Error] \n",
                  tcmMemoryTypeName[cfg.memType-1], retVal);
        goto EXIT;
    }

    MMW_PRINT("[SUCCESS] Diag %s Parity \n",  tcmMemoryTypeName[cfg.memType-1]);

EXIT:
    if (cfg.memType == CSL_RCM_MemoryType_TCMA)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_A_PARITY_TEST_STATUS_BIT);
    }
    else if (cfg.memType == CSL_RCM_MemoryType_TCMB0)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_B0_PARITY_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_TCM_B1_PARITY_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the ESM Static configuration
 *      diagnostic.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_EsmStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    uint32_t                    cycleCount = 0;

    /* Initialization ESM Baseline in Diag */
    retVal = Diag_ESM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag ESM Static Configuration [Init] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Compare the baseline & snapshot in Diag */
    retVal = Diag_ESM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT("[ERROR] Diag ESM Static Configuration [Compare] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT("[SUCCESS] Diag ESM Static Configuration \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_ESM_STATIC_TEST_STATUS_BIT);
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the ESM static configuration diagnostic once the steady
 *      state has been modified. The test will modify the ESM
 *      LTC preload value which changes the steady state.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_EsmStaticCfg_verifyViolationLTCPreload (Diag_StaticCfgErrInfo* ptrErrorInfo,
                                                         uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    CSL_ESMRegs*                ptrESMRegs;
    uint16_t                    value;
    uint16_t                    newValue;
    uint32_t                    cycleCount = 0;

    /* Get the ESM registers: */
    ptrESMRegs = CSL_ESM_getBaseAddress ();

    /* Initialize the diagnostic */
    retVal = Diag_ESM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag ESM StaticCfg Verify Violation LTC Preload [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Violate the steady state by modifying the LTC preload value
     ********************************************************************************/
    /* read the current preload value, change it and write it back */
    value = CSL_ESM_getLTCPreload(ptrESMRegs);
    newValue = (value & 0xC000U) ^ 0xC000U;
    CSL_ESM_setLTCPreload(ptrESMRegs, newValue);

    /* Execute the diagnostic: This should FAIL. */
    retVal = Diag_ESM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag ESM StaticCfg Verify Violation LTC Preload [Post steady state change] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The LTCPR register was modified and that is the offset that should be returned */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_ESMRegs, ESMLTCPR))
    {
        MMW_PRINT ("[ERROR] Diag ESM StaticCfg Verify Violation LTC Preload [Offset mismatch] \n");
        goto EXIT;
    }

    /* Reset the diagnostic: The new LTC Preload configuration is now in the baseline. */
    retVal = Diag_ESM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag ESM StaticCfg Verify Violation LTC Preload [Reset] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Update the steady state by restoring the LTC preload value to original value */
    CSL_ESM_setLTCPreload(ptrESMRegs, value);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag ESM StaticCfg Verify Violation LTC Preload \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_ESM_STATIC_TEST_STATUS_BIT);
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      This is the implementation of the RCM Static configuration
 *      diagnostic.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_RCMStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    uint32_t                    cycleCount = 0;

    /********************************************************************
     * Execute Initialization Diagnostic
     ********************************************************************/
    retVal = Diag_RCM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag RCM Static Configuration [Init] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Execute Compare Diagnostic
     ********************************************************************/
    retVal = Diag_RCM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag RCM Static Configuration [Compare] [%d]\n", retVal);
        MMW_PRINT("ErrorInfo: errorOffset=0x%x, baseline=0x%x, snapshot=0x%x\n",
                         ptrErrorInfo->errorOffset, ptrErrorInfo->baseline, ptrErrorInfo->snapshot);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Execute Reset Diagnostic
     ********************************************************************/
    retVal = Diag_RCM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag RCM Static Configuration [Reset] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag RCM StaticCfg Configuration \n");
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_RCM_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the RCM static configuration diagnostic once the steady
 *      state has been modified. The test will modify the RCM 
 *      ECCENMSSBSS register.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 *
 */
int32_t MssDiag_rcmStaticCfg_verifyViolation(Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    CSL_RCMRegs*                ptrRCMRegs;
    uint8_t                     value;
    uint8_t                     newValue;
    uint32_t                    cycleCount = 0;
    
    /* Get the RCM registers: */
    ptrRCMRegs = CSL_RCM_getBaseAddress ();

    /********************************************************************************
     * Initialize the diagnostic:
     ********************************************************************************/
    retVal = Diag_RCM_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag RCM StaticCfg Verify Violation Mailbox ECC [Init] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Violate the steady state by modifying the ECC for MSS-BSS mailbox
     ********************************************************************************/
    /* read the value, change it and write it back */
    value = CSL_RCM_getMailboxECC(ptrRCMRegs, CSL_RCM_MemoryType_MAILBOX_MSS_TO_BSS);
    newValue = value ^ 0x1U;
    CSL_RCM_setMailboxECC(ptrRCMRegs, CSL_RCM_MemoryType_MAILBOX_MSS_TO_BSS, newValue);

    /********************************************************************************
     * Execute the diagnostic:
     * - This should fail.
     ********************************************************************************/
    retVal = Diag_RCM_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag RCM StaticCfg Verify Violation Mailbox ECC [Post steady state change] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The ECCENMSSBSS register was modified and that is the offset that should be returned */
    if (ptrErrorInfo->errorOffset != offsetof (CSL_RCMRegs, ECCENMSSBSS))
    {
        MMW_PRINT ("[ERROR] Diag RCM StaticCfg Verify Violation Mailbox ECC [Offset mismatch] [%d]\n", retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Update the steady state by restoring the Mailbox ECC value to original value
     ********************************************************************************/
    CSL_RCM_setMailboxECC(ptrRCMRegs, CSL_RCM_MemoryType_MAILBOX_MSS_TO_BSS, value);

    /********************************************************************************
     * Reset the diagnostic:
     * - The Mailbox ECC is now restored to original value and is in the baseline.
     ********************************************************************************/
    retVal = Diag_RCM_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag RCM StaticCfg Verify Violation Mailbox ECC [Reset] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    MMW_PRINT ("[SUCCESS] Diag RCM StaticCfg Verify Violation Mailbox ECC \n");

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_RCM_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the R4F system Static configuration
 *      diagnostic.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_r4fStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    uint32_t                    cycleCount = 0;

    /********************************************************************
     * Diagnostic Initialization
     ********************************************************************/
    retVal = Diag_R4F_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F Static Configuration [Init] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Execute Diag COMPARE
     ********************************************************************/
    retVal = Diag_R4F_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F Static Configuration [Compare] [%d]\n", retVal);
        MMW_PRINT("ErrorInfo: errorOffset=0x%x, baseline=0x%x, snapshot=0x%x\n", \
                 ptrErrorInfo->errorOffset, ptrErrorInfo->baseline, ptrErrorInfo->snapshot);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************
     * Execute Diag RESET
     ********************************************************************/
    retVal = Diag_R4F_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F Static Configuration [Reset] [%d]\n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag R4F StaticCfg Configuration \n", *ptrCycleCount);
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_R4F_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to verify the mismatch of data for
 *      the R4F static configuration diagnostic once the steady
 *      state has been modified.
 *
 *  Note: This is a negative test case to test out the error path.
 *        In real use case the change of system control registers
 *        could result in unknown consequences.
 *
 *  @param[out] ptrErrorInfo
 *      This is populated with the error information if the diagnostic fails
 *      only with the error code set to DIAG_EDATA. The information specified
 *      here will indicate the error where the first mismatch was detected
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_r4fStaticCfg_verifyViolation (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t                     retVal;
    uint32_t                    expectedOffset;
    uint8_t                     value;
    uint8_t                     newValue;
    uint32_t                    cycleCount = 0;

    /********************************************************************************
     * Initialize the diagnostic:
     ********************************************************************************/
    retVal = Diag_R4F_staticCfgExecute (Diag_StaticCfgMode_INIT, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F StaticCfg Verify Violation TCMB1 ext error status [Init] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /********************************************************************************
     * Violate the steady state by modifying the TCMB1 external error status
     ********************************************************************************/
    /* read the value, change it and write it back */
    value = CSL_R4F_getTCMB1ExternalErrorStatus();
    newValue = value ^ 0x1U;
    CSL_R4F_setTCMB1ExternalErrorStatus(newValue);

    /********************************************************************************
     * Execute the diagnostic:
     * - This should fail.
     ********************************************************************************/
    retVal = Diag_R4F_staticCfgExecute (Diag_StaticCfgMode_COMPARE, ptrErrorInfo, ptrCycleCount);
    if (retVal == DIAG_SUCCESS)
    {
        MMW_PRINT ("[ERROR] Diag R4F StaticCfg Verify Violation TCMB1 ext error status [Post steady state change] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* Validate the error information:
     * - The AUXILIARY_CONTROL register was modified and that is the offset that should be returned */
     expectedOffset = offsetof(CSL_R4F_staticCfgRegs, AUXILIARY_CONTROL);
    if (ptrErrorInfo->errorOffset != expectedOffset)
    {
        MMW_PRINT("Offset mismatch: ptrErrorInfo->errorOffset=%d, expectedOffset=%d", ptrErrorInfo->errorOffset, expectedOffset);
        MMW_PRINT ("[ERROR] Diag R4F StaticCfg Verify Violation TCMB1 ext error status [Offset mismatch] \n", retVal);
        goto EXIT;
    }

    /********************************************************************************
     * Update the steady state by restoring the TCMB1 ext error status value to original value
     ********************************************************************************/
    CSL_R4F_setTCMB1ExternalErrorStatus(value);

    /********************************************************************************
     * Reset the diagnostic:
     * - The TCMB1 ext error status is now restored to original value and is in the baseline.
     ********************************************************************************/
    retVal = Diag_R4F_staticCfgExecute (Diag_StaticCfgMode_RESET, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F StaticCfg Verify Violation TCMB1 ext error status [Reset] [%d] \n", retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);
    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag R4F StaticCfg Verify Violation TCMB1 ext error status \n");

EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, DIAG_MSS_R4F_STATIC_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the R4F CCM diagnostic.
 *
 *  @param[in] cfg
 *      configuration used to execute the diagnostic.
 *  @param[in] ptrErrorInfo
 *      Pointer to the error information. This is only used if there is an
 *      error.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_r4fCcmTest (Diag_CCM_Cfg  cfg, Diag_CCM_ErrorInfo *ptrErrorInfo,
                            uint32_t *ptrCycleCount)
{
    int32_t              retVal;

    /* Execute the diagnostic. */
    retVal = Diag_CCM_execute(&cfg, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F CCM-%c Test [%d]\n", (65+cfg.instanceId), retVal);
        goto EXIT;
    }
    MMW_PRINT ("[SUCCESS] Diag R4F CCM-%c Test \n", (65+cfg.instanceId));
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_CCMA_TEST_STATUS_BIT+cfg.instanceId));
    return retVal;
}

/**
 *  @b Description
 *  @n
 *     The function is used to test the R4F DCC Diagnostic
 *      with basic configuration.
 *
 *  @param[in] cfg
 *      configuration used to execute the diagnostic.
 *  @param[in] ptrErrorInfo
 *      Pointer to the error information. This is only used if there is an
 *      error.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_dccDiagTest(Diag_DCC_Cfg cfg,
                            Diag_DCC_ErrorInfo *ptrErrorInfo, uint32_t* ptrCycleCount)
{
    int32_t              retVal;

    /* Execute the diagnostic. */
    retVal = Diag_DCC_execute(&cfg, ptrErrorInfo, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F DCC-%c [Clk1 freq as expected] \n", (65+cfg.instanceId), retVal);
        goto EXIT;
    }
    MMW_PRINT ("[SUCCESS] Diag R4F CCM-%c Negative Test \n", (65+cfg.instanceId));
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_DCCA_TEST_STATUS_BIT+cfg.instanceId));
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation negative DCC diagnostic test
 *
 *  @param[in] instanceId
 *      Instance ID for DCC
 *  @param[in] ptrErrorInfo
 *      Pointer to the error information. This is only used if there is an
 *      error.
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t MssDiag_dccViolationTest(uint8_t instanceId,
                                 Diag_DCC_ErrorInfo *ptrErrorInfo,
                                 uint32_t* ptrCycleCount)
{
    Diag_DCC_Cfg         cfg;
    int32_t              retVal;
    uint32_t             cycleCount = 0;

    (void)memset((void *)&cfg, 0, sizeof(Diag_DCC_Cfg));

    /***************************************************************************
     * Diagnostic: DCC-A/B Fail case with Clk1 faster than expected
     ***************************************************************************/
    /* Populate the configuration. */
    cfg.instanceId      = instanceId;
    if(instanceId == 0)
    {
        cfg.clkSrc0         = DCCA_CLK_SRC0;
        cfg.clkSrc1         = DCCA_CLK_SRC1;
        cfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
        cfg.countSeed0      = DCCA_COUNTSEED0 + (DCCA_VALIDSEED0 * 2);
        cfg.countSeed1      = DCCA_COUNTSEED1;
        cfg.validSeed0      = DCCA_VALIDSEED0;
        cfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT;
    }
    else
    {
        cfg.clkSrc0         = DCCB_CLK_SRC0;
        cfg.clkSrc1         = DCCB_CLK_SRC1;
        cfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
        cfg.countSeed0      = DCCB_COUNTSEED0 + (DCCB_VALIDSEED0 * 2);
        cfg.countSeed1      = DCCB_COUNTSEED1;
        cfg.validSeed0      = DCCB_VALIDSEED0;
        cfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT;
    }
    /* Execute the diagnostic. */
    retVal = Diag_DCC_execute(&cfg, ptrErrorInfo, ptrCycleCount);
    /* Did we detect that the clk1 is faster. If not test failed */
    if ((ptrErrorInfo->clkSrc1FreqStatus != Diag_DCC_CLKSRC1_FREQ_FASTER) ||\
        (retVal != DIAG_SUCCESS))
    {
        retVal = DIAG_SUCCESS;
    }
    else
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F DCC-%c [Clk1 freq faster] [%d] \n", (65+instanceId), retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: DCCA Fail case with Clk1 slower than expected
     ***************************************************************************/
    /* Populate the configuration. */
    cfg.instanceId      = instanceId;
    if(instanceId == 0)
    {
        cfg.clkSrc0         = DCCA_CLK_SRC0;
        cfg.clkSrc1         = DCCA_CLK_SRC1;
        cfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
        cfg.countSeed0      = DCCA_COUNTSEED0 - (DCCA_VALIDSEED0 * 2);
        cfg.countSeed1      = DCCA_COUNTSEED1;
        cfg.validSeed0      = DCCA_VALIDSEED0;
        cfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT;
    }
    else
    {
        cfg.clkSrc0         = DCCB_CLK_SRC0;
        cfg.clkSrc1         = DCCB_CLK_SRC1;
        cfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
        cfg.countSeed0      = DCCB_COUNTSEED0 - (DCCB_VALIDSEED0 * 2);
        cfg.countSeed1      = DCCB_COUNTSEED1;
        cfg.validSeed0      = DCCB_VALIDSEED0;
        cfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT;
    }
    /* Execute the diagnostic. */
    retVal = Diag_DCC_execute(&cfg, ptrErrorInfo, ptrCycleCount);

    /* Did we detect that the clk1 is slower. If not test failed */
    if ((ptrErrorInfo->clkSrc1FreqStatus != Diag_DCC_CLKSRC1_FREQ_SLOWER) ||\
        (retVal != DIAG_SUCCESS))
    {
        retVal = DIAG_SUCCESS;
    }
    else
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F DCC-%c [Clk1 freq Slower] [%d]\n", (65+instanceId), retVal);
        goto EXIT;
    }
    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /***************************************************************************
     * Diagnostic: DCCA Fail case Continuous mode
     ***************************************************************************/
    (void)memset((void *)&cfg, 0, sizeof(Diag_DCC_Cfg));

    /* Populate the configuration. */
    cfg.instanceId      = instanceId;
    if(instanceId == 0)
    {
        cfg.clkSrc0         = DCCA_CLK_SRC0;
        cfg.clkSrc1         = DCCA_CLK_SRC1;
        cfg.mode            = Diag_CLKCOMPARATOR_MODE_CONTINUOUS;
        cfg.countSeed0      = DCCA_COUNTSEED0 + (DCCA_VALIDSEED0 * 2);
        cfg.countSeed1      = DCCA_COUNTSEED1;
        cfg.validSeed0      = DCCA_VALIDSEED0;
        cfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE;
    }
    else
    {
        cfg.clkSrc0         = DCCB_CLK_SRC0;
        cfg.clkSrc1         = DCCB_CLK_SRC1;
        cfg.mode            = Diag_CLKCOMPARATOR_MODE_CONTINUOUS;
        cfg.countSeed0      = DCCB_COUNTSEED0 + (DCCB_VALIDSEED0 * 2);
        cfg.countSeed1      = DCCB_COUNTSEED1;
        cfg.validSeed0      = DCCB_VALIDSEED0;
        cfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE;
    }
    /* Execute the diagnostic. */
    retVal = Diag_DCC_execute(&cfg, ptrErrorInfo, ptrCycleCount);

    /* Did we detect that the clk1 is faster. If not test failed */
    if ((ptrErrorInfo->clkSrc1FreqStatus != Diag_DCC_CLKSRC1_FREQ_FASTER) ||\
        (retVal != DIAG_SUCCESS))
    {
        retVal = DIAG_SUCCESS;
    }
    else
    {
        /*  Test Result Reporting: */
        MMW_PRINT ("[ERROR] Diag R4F DCC-%c [Continuous mode - Clk1 freq faster] [%d]\n", (65+instanceId), retVal);
        goto EXIT;
    }

    /* update the cyclecount if requested by the func caller */
    UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);
    MMW_PRINT ("[SUCCESS] Diag R4F DCC-%c Test \n", (65+instanceId));
EXIT:
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gMssDiagTestStatus, (DIAG_MSS_DCCA_TEST_STATUS_BIT+instanceId));

    return retVal;
}


/**
 *  @b Description
 *  @n
 *      Print which MSS boot time test(s) failed.
 *
 *  @retval
 *      Not applicable
 */
void MssDiag_BootStatusPrintErrorInfo(const Diag_MSSBootTest_Status* ptrStatus)
{
   /* Check ESMGroup1Status */
    if (ptrStatus->ESMGroup1Status != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_ESM_GRP1_ERROR;
        gDiagErrDetail.errCauseInfo = ptrStatus->ESMGroup1Status;
    }

    /* Check MIBSPIStatus */
    if (ptrStatus->MIBSPIStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MIBSPI_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check MIBSPIRAMSBStatus */
    if (ptrStatus->MIBSPIRAMSBStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MIBSPI_RAM_SINGLE_BIT_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check MIBSPIRAMDBStatus */
    if (ptrStatus->MIBSPIRAMDBStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MIBSPI_RAM_DOUBLE_BIT_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check PBISTSinglePortMemStatus */
    if (ptrStatus->PBISTSinglePortMemStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_PBIST_SINGLE_PORT_MEM_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check PBISTDualPortMemStatus */
    if (ptrStatus->PBISTDualPortMemStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_PBIST_DUAL_PORT_MEM_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check MemInitStatus */
    if (ptrStatus->MemInitStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MEM_INIT_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check RTIStatus */
    if (ptrStatus->RTIStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_RTI_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check ROMCRCStatus */
    if (ptrStatus->ROMCRCStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_ROM_CRC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check CRCStatus */
    if (ptrStatus->CRCStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MSS_CRC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check MPUStatus */
    if (ptrStatus->MPUStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MPU_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check ESMStatus */
    if (ptrStatus->ESMStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_ESM_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check DMAStatus */
    if (ptrStatus->DMAStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_DMA_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check DMARAMParityStatus */
    if (ptrStatus->DMARAMParityStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_DMA_RAM_PARITY_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check DMAMPUStatus */
    if (ptrStatus->DMAMPUStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_DMA_MPU_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check VIMStatus */
    if (ptrStatus->VIMStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_VIM_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check DCCStatus */
    if (ptrStatus->DCCStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_DCC_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check DCCFaultInsertionStatus */
    if (ptrStatus->DCCFaultInsertionStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_DCC_FAULT_INSR_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check PCRFaultInsertionStatus */
    if (ptrStatus->PCRFaultInsertionStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_PCR_FAULT_INSR_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check VIMRAMParityStatus */
    if (ptrStatus->VIMRAMParityStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_VIM_RAM_PARITY_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check UARTStatus */
    if (ptrStatus->UARTStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_UART_FUNC_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

#ifdef SOC_XWR68XX

    /* Check MSSLBISTStatus */
    if (ptrStatus->MSSLBISTStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_MSS_LBIST_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

    /* Check GEMLBISTPBISTStatus */
    if (ptrStatus->GEMLBISTPBISTStatus != 0U)
    {
        gDiagErrDetail.errCode = DIAG_BOOT_DSS_LBIST_PBIST_TEST_ERROR;
        gDiagErrDetail.errCauseInfo = 0;
    }

#endif /* SOC_XWR68XX */

    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(gDiagErrDetail.errCode, gMssDiagTestStatus, DIAG_MSS_DCCB_TEST_STATUS_BIT);
    MMW_PRINT("[ERROR] BootROM Test Status [%d]", gDiagErrDetail.errCode);
}


/**
 *  @b Description
 *  @n
 *      This is the implementation of the diagnostic that returns the status of various
 *      boot time tests run by MSS bootrom.
 *
 *  @param[out] ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information. If this
 *      is set to NULL then the benchmarking information is not
 *      reported back to the application.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS (all boot time tests passed)
 *  @Note
 *       BooROM test status is already being verified by the SBL,
 *       provided here for reference purpose.
 */
int32_t MssDiag_bootTestStatus (uint32_t* ptrCycleCount)
{
    Diag_MSSBootTest_Status  bootTestStatus;
    int32_t                  retVal;

    /***************************************************************************
     * Diagnostic: Get boot test status
     ***************************************************************************/
    (void)memset((void *)&bootTestStatus, 0, sizeof(Diag_MSSBootTest_Status));

    /* Execute the diagnostic. */
    retVal = Diag_getMSSBootTestStatus(&bootTestStatus, ptrCycleCount);
    
    if (retVal == DIAG_EDATA)
    {
        MssDiag_BootStatusPrintErrorInfo(&bootTestStatus);
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of all the MSS Self Test diagnostic Tests.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */ 
int32_t MssDiag_SelfTest(void)
{
    int32_t         retVal;
    uint32_t        cycleCount = 0;
    Diag_CCM_Cfg       ccmCfg;
    Diag_CCM_ErrorInfo ccmErrInfo;
    Diag_DCC_ErrorInfo dccErrInfo;
    Diag_DCC_Cfg       dccCfg;

    memset(&ccmCfg, 0, sizeof(Diag_CCM_Cfg));
    memset(&dccCfg, 0, sizeof(Diag_DCC_Cfg));
    memset(&ccmErrInfo, 0, sizeof(Diag_CCM_ErrorInfo));
    memset(&dccErrInfo, 0, sizeof(Diag_DCC_ErrorInfo));

    /**** Self test diagnostics ***/
    /* Single Shot CCCA Single Mode Diagnostic Test */
    ccmCfg.instanceId      = 0U;
    ccmCfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
    ccmCfg.clkSrc0         = CCCA_CLK_SRC0;
    ccmCfg.clkSrc1         = CCCA_CLK_SRC1;
    ccmCfg.count0Expiry    = CCCA_COUNT0_EXPIRY;
    ccmCfg.count1Expected  = CCCA_COUNT1_EXPECTED;
    ccmCfg.marginCount     = CCCA_MARGIN_COUNT;
    ccmCfg.count1Error     = CCCA_COUNT1_ERROR;
    ccmCfg.maxCycleCount   = CCC_MAX_CYCLE_COUNT;
    retVal = MssDiag_r4fCcmTest(ccmCfg, &ccmErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* Continuous Shot CCCA Continuous Mode Diagnostic Test */
    ccmCfg.instanceId      = 0U;
    ccmCfg.mode            = Diag_CLKCOMPARATOR_MODE_CONTINUOUS;
    ccmCfg.clkSrc0         = CCCA_CLK_SRC0;
    ccmCfg.clkSrc1         = CCCA_CLK_SRC1;
    ccmCfg.count0Expiry    = CCCA_COUNT0_EXPIRY;
    ccmCfg.count1Expected  = CCCA_COUNT1_EXPECTED;
    ccmCfg.marginCount     = CCCA_MARGIN_COUNT;
    ccmCfg.count1Error     = CCCA_COUNT1_ERROR;
    ccmCfg.maxCycleCount   = CCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE;
    retVal = MssDiag_r4fCcmTest(ccmCfg, &ccmErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* Single Shot CCC-B Single Mode Diagnostic Test */
    ccmCfg.instanceId      = 1U;
    ccmCfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
    ccmCfg.clkSrc0         = CCCB_CLK_SRC0;
    ccmCfg.clkSrc1         = CCCB_CLK_SRC1;
    ccmCfg.count0Expiry    = CCCB_COUNT0_EXPIRY;
    ccmCfg.count1Expected  = CCCB_COUNT1_EXPECTED;
    ccmCfg.marginCount     = CCCB_MARGIN_COUNT;
    ccmCfg.count1Error     = CCCB_COUNT1_ERROR;
    ccmCfg.maxCycleCount   = CCC_MAX_CYCLE_COUNT;
    retVal = MssDiag_r4fCcmTest(ccmCfg, &ccmErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* Continuous Shot CCC-B Continuous Mode Diagnostic Test */
    ccmCfg.instanceId      = 1U;
    ccmCfg.mode            = Diag_CLKCOMPARATOR_MODE_CONTINUOUS;
    ccmCfg.clkSrc0         = CCCB_CLK_SRC0;
    ccmCfg.clkSrc1         = CCCB_CLK_SRC1;
    ccmCfg.count0Expiry    = CCCB_COUNT0_EXPIRY;
    ccmCfg.count1Expected  = CCCB_COUNT1_EXPECTED;
    ccmCfg.marginCount     = CCCB_MARGIN_COUNT;
    ccmCfg.count1Error     = CCCB_COUNT1_ERROR;
    ccmCfg.maxCycleCount   = CCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE;
    retVal = MssDiag_r4fCcmTest(ccmCfg, &ccmErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* R4F DCC-A Single Mode diagnostic. */
    /* Populate the configuration. */
    dccCfg.instanceId      = 0U;
    dccCfg.clkSrc0         = DCCA_CLK_SRC0;
    dccCfg.clkSrc1         = DCCA_CLK_SRC1;
    dccCfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
    dccCfg.countSeed0      = DCCA_COUNTSEED0;
    dccCfg.countSeed1      = DCCA_COUNTSEED1;
    dccCfg.validSeed0      = DCCA_VALIDSEED0;
    dccCfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT;
    retVal = MssDiag_dccDiagTest(dccCfg, &dccErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* R4F DCC-A Continuous Mode diagnostic. */
    /* Populate the configuration. */
    dccCfg.instanceId      = 0U;
    dccCfg.clkSrc0         = DCCA_CLK_SRC0;
    dccCfg.clkSrc1         = DCCA_CLK_SRC1;
    dccCfg.mode            = Diag_CLKCOMPARATOR_MODE_CONTINUOUS;
    dccCfg.countSeed0      = DCCA_COUNTSEED0;
    dccCfg.countSeed1      = DCCA_COUNTSEED1;
    dccCfg.validSeed0      = DCCA_VALIDSEED0;
    dccCfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE;
    retVal = MssDiag_dccDiagTest(dccCfg, &dccErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* R4F DCC-B Single Mode diagnostic. */
    /* Populate the configuration. */
    dccCfg.instanceId      = 1U;
    dccCfg.clkSrc0         = DCCB_CLK_SRC0;
    dccCfg.clkSrc1         = DCCB_CLK_SRC1;
    dccCfg.mode            = Diag_CLKCOMPARATOR_MODE_SINGLESHOT;
    dccCfg.countSeed0      = DCCB_COUNTSEED0;
    dccCfg.countSeed1      = DCCB_COUNTSEED1;
    dccCfg.validSeed0      = DCCB_VALIDSEED0;
    dccCfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT;
    retVal = MssDiag_dccDiagTest(dccCfg, &dccErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* R4F DCC-B Continuous Mode diagnostic. */
    /* Populate the configuration. */
    dccCfg.instanceId      = 1U;
    dccCfg.clkSrc0         = DCCB_CLK_SRC0;
    dccCfg.clkSrc1         = DCCB_CLK_SRC1;
    dccCfg.mode            = Diag_CLKCOMPARATOR_MODE_CONTINUOUS;
    dccCfg.countSeed0      = DCCB_COUNTSEED0;
    dccCfg.countSeed1      = DCCB_COUNTSEED1;
    dccCfg.validSeed0      = DCCB_VALIDSEED0;
    dccCfg.maxCycleCount   = DCC_MAX_CYCLE_COUNT_CONTINUOUS_MODE;
    retVal = MssDiag_dccDiagTest(dccCfg, &dccErrInfo, &cycleCount);
    ASSERT(retVal == 0);
    return retVal;
}


int32_t MssDiag_ErrorInjectTest(void)
{
    int32_t retVal = 0;
    Diag_StaticCfgErrInfo       errorInfo;
    Diag_Mailbox_ECC_ErrorInfo  mbEccErrInfo;
    Diag_MIBSPI_ECC_ErrorInfo   mibspiEccErrInfo;
    Diag_VIM_ECC_Cfg            vimEccCfg;
    Diag_TCM_Parity_Cfg         tcmParityCfg;
    Diag_TCM_ECC_Cfg            tcmEccCfg;
    Diag_DMA_MPU_Cfg            dmaMpuCfg;
    Diag_DMA_Parity_Cfg         dmaParityCfg;
    Diag_MIBSPI_ECC_Cfg         mibspiEccCfg;
    Diag_Mailbox_ECC_Cfg        mailboxEccCfg;
    Diag_CCM_ErrorInfo          ccmErrInfo;
    uint8_t                      mbox;
    uint8_t                      mode;
    uint32_t                    offset = 0;
    uint32_t                    cycleCount = 0;

    memset(&errorInfo, 0, sizeof(Diag_StaticCfgErrInfo));
    memset(&mbEccErrInfo, 0, sizeof(Diag_Mailbox_ECC_ErrorInfo));
    memset(&mibspiEccErrInfo, 0, sizeof(Diag_MIBSPI_ECC_ErrorInfo));
    memset(&vimEccCfg, 0, sizeof(Diag_VIM_ECC_Cfg));
    memset(&tcmParityCfg, 0, sizeof(Diag_TCM_Parity_Cfg));
    memset(&dmaMpuCfg, 0, sizeof(Diag_DMA_MPU_Cfg));
    memset(&dmaParityCfg, 0, sizeof(Diag_DMA_Parity_Cfg));
    memset(&mibspiEccCfg, 0, sizeof(Diag_MIBSPI_ECC_Cfg));
    memset(&mailboxEccCfg, 0, sizeof(Diag_Mailbox_ECC_Cfg));

    memset(&ccmErrInfo, 0, sizeof(Diag_CCM_ErrorInfo));
    memset(&tcmEccCfg, 0, sizeof(Diag_TCM_ECC_Cfg));

    /******** VIM RAM ECC diagnostic [ERROR INJECTION]  *******/
    /* single Bit VIM ECC error injection Test */
    vimEccCfg.eccMode       = Diag_ECCMode_SINGLE_BIT_ERROR;
    vimEccCfg.maxCycleCount = 1000U;
    retVal = MssDiag_vimEccTest(vimEccCfg, &cycleCount);
    ASSERT(retVal == 0);

       /* Double Bit VIM ECC error injection Test */
    vimEccCfg.eccMode       = Diag_ECCMode_DOUBLE_BIT_ERROR;
    vimEccCfg.maxCycleCount = 1000U;
    retVal = MssDiag_vimEccTest(vimEccCfg, &cycleCount);
    ASSERT(retVal == 0);

    /******** Mailbox diagnostic [ERROR INJECTION]  *******/
    /* MailbBox ECC Error Injection Diagnostic Test.
     * Loop through each mailbox */
    for (mbox = 0U; mbox < NUM_MAILBOX_MEMORY_TYPE; mbox++)
    {
        /* Loop through each ECC type */
        for (mode = Diag_ECCMode_SINGLE_BIT_ERROR;
             mode <= Diag_ECCMode_DOUBLE_BIT_ERROR; mode++)
        {
            /* Populate the configuration. */
            mailboxEccCfg.eccMode         = (Diag_ECCMode)mode;
            mailboxEccCfg.memType         = mailboxMemoryType[mbox];
            mailboxEccCfg.addrOffset      = offset;
            mailboxEccCfg.maxCycleCount   = MAILBOX_ECC_MAX_CYCLE_COUNT;

            retVal = MssDiag_MailboxEccTest(mailboxEccCfg, &mbEccErrInfo, &cycleCount);
            ASSERT(retVal == 0);
            offset = offset + 16U; /* Offset has to be 8 byte aligned */
        }
    }

    /*** TCM ECC single diagnostic [ERROR INJECTION]  ***/
    /* TCMA Single BIT ECC Diag test */
    tcmEccCfg.memType       = CSL_RCM_MemoryType_TCMA;
    tcmEccCfg.eccMode       = Diag_ECCMode_SINGLE_BIT_ERROR;
    tcmEccCfg.maxCycleCount = 100000U;
    retVal = MssDiag_TcmEccTest(tcmEccCfg, &cycleCount);
    ASSERT(retVal == 0);

    /******** TCMA, TCMB0, TCMB1 Parity diagnostic [ERROR INJECTION]  *******/
    /* TCMA Parity Error injection Test */
    tcmParityCfg.memType       = CSL_RCM_MemoryType_TCMA;
    tcmParityCfg.maxCycleCount = 1000U;
    retVal = MssDiag_TcmParityTest(tcmParityCfg, &cycleCount);
    ASSERT(retVal == 0);
    /* TCMB0 Parity Error injection Test */
    tcmParityCfg.memType       = CSL_RCM_MemoryType_TCMB0;
    tcmParityCfg.maxCycleCount = 1000U;
    retVal = MssDiag_TcmParityTest(tcmParityCfg, &cycleCount);
    ASSERT(retVal == 0);
    /* TCMB1 Parity Error injection Test */
    tcmParityCfg.memType       = CSL_RCM_MemoryType_TCMB1;
    tcmParityCfg.maxCycleCount = 1000U;
    retVal = MssDiag_TcmParityTest(tcmParityCfg, &cycleCount);
    ASSERT(retVal == 0);


    /******** DMA instance 0 & 1 MPU diagnostic [ERROR INJECTION]  *******/
    /* DMA instance 0: MPU error Injection Test for all the channels.
     * Populate the diagnostic configuration: */
    dmaMpuCfg.dmaInstanceId      = 0;
    dmaMpuCfg.regionSize         = 4U;
    dmaMpuCfg.maxCycleCount      = 200U;
    /* These parameters can be within the test for different range of values */
    dmaMpuCfg.dmaChannel         = DMA_ERR_INJECT_CHAN;
    dmaMpuCfg.mpuRegion          = DMA_MPU_REGION_IDX;
    /* This parameter is being set within Test for all 4 memory regions
     dmaMpuCfg.regionStartAddress = (uint32_t)memoryRegionCfg[memoryRegionIndex];
     */
    /* error Injection DMA MPU Diag Test */
    retVal = MssDiag_dmaMpuTest(dmaMpuCfg, &cycleCount);
    ASSERT(retVal == 0);

    /* DMA instance 1: MPU error Injection Test for all the channels.
     * Populate the diagnostic configuration: */
    dmaMpuCfg.dmaInstanceId      = 1;
    dmaMpuCfg.regionSize         = 4U;
    dmaMpuCfg.maxCycleCount      = 200U;
    /* These parameters can be within the test for different range of values */
     dmaMpuCfg.dmaChannel         = DMA_ERR_INJECT_CHAN;
     dmaMpuCfg.mpuRegion          = DMA_MPU_REGION_IDX;
     /* This parameter is being set within Test for all 4 memory regions
       dmaMpuCfg.regionStartAddress = (uint32_t)memoryRegionCfg[memoryRegionIndex];
      */
    /* error Injection DMA MPU Diag Test */
    retVal = MssDiag_dmaMpuTest(dmaMpuCfg, &cycleCount);
    ASSERT(retVal == 0);

    /******** DMA instance 0 & 1 Parity diagnostic [ERROR INJECTION]  *******/
    /* DMA Instance 0 Parity Error Injection Test for selected channel(s) */
    dmaParityCfg.dmaInstanceId = 0;
    dmaParityCfg.maxCycleCount = 2000U;
    /* This parameter can be within the test for different range of values */
    dmaParityCfg.dmaChannel    = DMA_ERR_INJECT_CHAN;
    retVal = MssDiag_dmaParityTest(dmaParityCfg, &cycleCount);
    ASSERT(retVal == 0);

    /* DMA Instance 1 Parity Error Injection Test for selected channel(s) */
    dmaParityCfg.dmaInstanceId = 1;
    dmaParityCfg.maxCycleCount = 2000U;
    /* This parameter can be within the test for different range of values */
    dmaParityCfg.dmaChannel    = DMA_ERR_INJECT_CHAN;
    retVal = MssDiag_dmaParityTest(dmaParityCfg, &cycleCount);
    ASSERT(retVal == 0);

    /**** MibSPI instance 0 & 1 ECC Single/Double diagnostic [ERROR INJECTION]  ****/
    /* MibSPI RAM ECC Self injection Test.
     * Select MibSPI Instance ID and ECC bit type: single/double.
     * This test runs requested type of ECC on RX & TX RAM of that instance */
    /* MibSPI Instance 0, Single Bit ECC Error injection Test */
    mibspiEccCfg.instanceId      = 0U;
    mibspiEccCfg.eccMode         = Diag_ECCMode_SINGLE_BIT_ERROR;
    /* this test function internally runs on Rx & TX both RAMs */
    mibspiEccCfg.ramBufferType   = CSL_MIBSPI_RamType_TX;
    mibspiEccCfg.ramBufferIndex  = MIBSPI_RAM_BUFFER_IDX_ECC_INJECT;
    mibspiEccCfg.maxCycleCount   = 1000U;
    retVal = MssDiag_MibspiEccTest(mibspiEccCfg, &mibspiEccErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* MibSPI Instance 0, double Bit ECC Error injection Test */
    mibspiEccCfg.instanceId      = 0U;
    mibspiEccCfg.eccMode         = Diag_ECCMode_DOUBLE_BIT_ERROR;
    /* this test function internally runs on Rx & TX both RAMs */
    mibspiEccCfg.ramBufferType   = CSL_MIBSPI_RamType_TX;
    mibspiEccCfg.ramBufferIndex  = MIBSPI_RAM_BUFFER_IDX_ECC_INJECT;
    mibspiEccCfg.maxCycleCount   = 1000U;
    retVal = MssDiag_MibspiEccTest(mibspiEccCfg, &mibspiEccErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* MibSPI Instance 1, Single Bit ECC Error injection Test */
    mibspiEccCfg.instanceId      = 1U;
    mibspiEccCfg.eccMode         = Diag_ECCMode_SINGLE_BIT_ERROR;
    /* this test function internally runs on Rx & TX both RAMs */
    mibspiEccCfg.ramBufferType   = CSL_MIBSPI_RamType_TX;
    mibspiEccCfg.ramBufferIndex  = MIBSPI_RAM_BUFFER_IDX_ECC_INJECT;
    mibspiEccCfg.maxCycleCount   = 1000U;
    retVal = MssDiag_MibspiEccTest(mibspiEccCfg, &mibspiEccErrInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* MibSPI Instance 1, double Bit ECC Error injection Test */
    mibspiEccCfg.instanceId      = 1U;
    mibspiEccCfg.eccMode         = Diag_ECCMode_DOUBLE_BIT_ERROR;
    /* this test function internally runs on Rx & TX both RAMs */
    mibspiEccCfg.ramBufferType   = CSL_MIBSPI_RamType_TX;
    mibspiEccCfg.ramBufferIndex  = MIBSPI_RAM_BUFFER_IDX_ECC_INJECT;
    mibspiEccCfg.maxCycleCount   = 1000U;
    retVal = MssDiag_MibspiEccTest(mibspiEccCfg, &mibspiEccErrInfo, &cycleCount);
    ASSERT(retVal == 0);


    /******** Watchdog diagnostic [ERROR INJECTION]  *******/
    retVal = MssDiag_WatchdogTest(&cycleCount);
    ASSERT(retVal == 0);

    return retVal;
}

int32_t MssDiag_StaticConfigTest(void)
{
    int32_t     retVal;
    uint32_t    cycleCount = 0;
    Diag_StaticCfgErrInfo errorInfo = {0};

    /************** Static Config Diagnostic Tests *******************/
    /* ESM Static config test */
    retVal = MssDiag_EsmStaticTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* VIM Static config test */
    retVal = MssDiag_vimStaticTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* DMA Static test : instance ID 0 */
    retVal = MssDiag_dmaStaticTest(0U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);
    /* DMA Static test : instance ID 1 */
    retVal = MssDiag_dmaStaticTest(1U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* MibSPI-A Static Config Diag test */
    retVal = MssDiag_MibspiStaticTest(0U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* MibSPI-B Static Config Diag test */
    retVal = MssDiag_MibspiStaticTest(1U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);
#ifndef SOC_XWR68XX
    /* DCAN  Static Config Diag test */
    retVal = MssDiag_DcanStaticTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);
#endif
    /* MCAN  Static Config Diag test */
    retVal = MssDiag_McanStaticTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* RCM Static Configuration Diagnostic Test */
    retVal = MssDiag_RCMStaticTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* R4F Static Configuration Diagnostic Test */
    retVal = MssDiag_r4fStaticTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* RTI Static configuration Diagnostic test */
    retVal = MssDiag_RtiStaticCfgTest(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /****** Negative Static Configuration Diagnostic Test *******/

    /* Diag DMA0 verify invalid MPU: negative test */
    retVal = MssDiag_dmaStaticCfg_verifyMPU(0U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);
    /* Diag DMA1 verify invalid MPU: negative test */
    retVal = MssDiag_dmaStaticCfg_verifyMPU(1U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* ESM Static config negative Test:
     * modify the ESM LTC preload value which changes the steady state. */
    retVal = MssDiag_EsmStaticCfg_verifyViolationLTCPreload(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* VIM Static Config negative Test:
     * modify the VIM wakeup interrupt configuration which changes the steady state. */
    retVal = MssDiag_vimStaticCfg_verifyViolationWakeup(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* VIM Static Config negative Test:
     * modify the VIM Fallback vector address which changes the steady state. */
    retVal = MssDiag_vimStaticCfg_verifyViolationFallbackAddr(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* VIM Static Config negative Test:
     * modify the VIM TEST_DIAG_EN bits in ECCCTL which changes the steady state. */
    retVal = MssDiag_vimStaticCfg_verifyViolationECCDiag(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* VIM Static Config Negative Test:
     * modify the VIM Interrupt Control Register (CHANCTRL) which changes the steady state.
     */
    retVal = MssDiag_vimStaticCfg_verifyViolationChanCtrl(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* MibSPI-A Static Config Violation Diag test */
    MssDiag_MibSPiStaticViolationTest(0U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);


    /* MibSPI-B Static Config Violation Diag test */
    MssDiag_MibSPiStaticViolationTest(0U, &errorInfo, &cycleCount);
    ASSERT(retVal == 0);

#ifndef SOC_XWR68XX
    /* DCAN  Static Config Violation Diag test */
    retVal = MssDiag_dcanStatic_verifyConfig(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);
#endif

    /* MCAN  Static Config Violation Diag test */
    retVal = MssDiag_mcanStatic_verifyConfig(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);


    /* RCM Static Configuration Violation Diagnostic Test */
    retVal = MssDiag_rcmStaticCfg_verifyViolation(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    /* R4F Static Configuration Violation Diagnostic Test */
    retVal = MssDiag_r4fStaticCfg_verifyViolation(&errorInfo, &cycleCount);
    ASSERT(retVal == 0);

    return retVal;
}

