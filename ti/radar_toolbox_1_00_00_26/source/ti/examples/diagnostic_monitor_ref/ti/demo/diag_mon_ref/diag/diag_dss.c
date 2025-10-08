/**
 *   @file  diag_dss.c
 *
 *   @brief
 *      This file contains all the SDL tests need to execute on DSS core.
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

/* mmWave SDL Include Files. */
#include <ti/csl/csl.h>
#include <ti/diag/diag.h>
#include <ti/diag/include/Diag_internal.h>
#include "osal/osal.h"
#include "diag/diag_test_api.h"
/* Diagnostic Error codes */
#include "diag/diag_error_code.h"

#define UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(a, b)  {a = ((b != NULL) ? (a + *b) : a);}
#define UPDATE_DIAG_TEST_CYCLE_COUNT(a, b)      if(a!=NULL){*a += b;}

#define L3_ECC_MAX_CYCLE_COUNT     100000U
#define L3_ECC_RESERVED_COUNT      16U

#define L2_PARITY_MAX_CYCLE_COUNT     100000U
#define L2_PARITY_RESERVED_COUNT      32U     /* Since L2 has 256 bit access, reserve 32 bytes */


#define L2_ECC_MAX_CYCLE_COUNT          100000U
#define L2_ECC_STRESS_LOOP_COUNT        10U
#define L2_ECC_IDMA_SINGLE_XFER_COUNT   0x8000U


#define L1P_PARITY_XFER_COUNT      32U     /* Has to be multiple of 32 bytes */
#define L1P_PARITY_MAX_CYCLE_COUNT 10000U

#define HSRAM_ECC_MAX_CYCLE_COUNT     100000U
#define HSRAM_ECC_RESERVED_COUNT      16U


#define DATA_TXFR_RAM_ECC_MAX_CYCLE_COUNT     100000U
#define DATA_TXFR_RAM_ECC_RESERVED_COUNT      16U

#define HWA_LOCKSTEP_MAX_CYCLE_COUNT     100000U
#define HWA_ECC_MAX_CYCLE_COUNT          100000U

/* EDMA TPCC Paramset selection for DIAG, user can change this if required */
#define TPCC0_PARAMSET_SELECTION                3
#define TPCC1_PARAMSET_SELECTION                2

#if 0 /* return as soon as any test fails */
#define RETURN_IF_ERROR(a)  if(!a){return a;}
#else /* don't return but proceed further test execution */
#define RETURN_IF_ERROR(a)
#endif

/**************************************************************************
 ************************** Platform Definitions **************************
 **************************************************************************/

#if defined (SOC_XWR16XX)
    #define EDMA_TPCC0_MAX_NUM_PARAM_SETS   CSL_XWR16XX_C674_EDMA_TPCC0_NUM_PARAM_SETS
    #define EDMA_TPCC1_MAX_NUM_PARAM_SETS   CSL_XWR16XX_C674_EDMA_TPCC1_NUM_PARAM_SETS
#elif defined (SOC_XWR18XX)
    #define EDMA_TPCC0_MAX_NUM_PARAM_SETS   CSL_XWR18XX_C674_EDMA_TPCC0_NUM_PARAM_SETS
    #define EDMA_TPCC1_MAX_NUM_PARAM_SETS   CSL_XWR18XX_C674_EDMA_TPCC1_NUM_PARAM_SETS
#elif defined (SOC_XWR68XX)
    #define EDMA_TPCC0_MAX_NUM_PARAM_SETS   CSL_XWR68XX_C674_EDMA_TPCC0_NUM_PARAM_SETS
    #define EDMA_TPCC1_MAX_NUM_PARAM_SETS   CSL_XWR68XX_C674_EDMA_TPCC1_NUM_PARAM_SETS
#else
    #error "Invalid SOC: Unable to build the unit tests"
#endif

extern void Diag_L2_ECC_setEDCMode(CSL_DSPICFGRegs* ptrDSPICFGRegs, uint8_t mode);


 /* Reserve memory in DATA_TXFR_RAM memory for DATA_TXFR_RAM ECC error injection */
 #pragma DATA_SECTION (DataTxfrRAMECCData, "dataTxfrRAMECC_diag_data");
 #pragma DATA_ALIGN ( DataTxfrRAMECCData , DATA_TXFR_RAM_ECC_RESERVED_COUNT )
 uint32_t DataTxfrRAMECCData[DATA_TXFR_RAM_ECC_RESERVED_COUNT/4U];

/* Reserve memory in HSRAM memory for HSRAM ECC error injection */
#pragma DATA_SECTION (HSRAMECCData, "HSRAMECC_diag_data");
#pragma DATA_ALIGN ( HSRAMECCData , HSRAM_ECC_RESERVED_COUNT )
uint32_t HSRAMECCData[HSRAM_ECC_RESERVED_COUNT/4U];

/* L1P_PARITY_XFER_COUNT bytes aligned to 32byte boundary reserved in L1P for L1P Parity diagnostics */
#pragma DATA_SECTION (L1PParityL1PData, "l1p_diag_data");
#pragma DATA_ALIGN ( L1PParityL1PData , L1P_PARITY_XFER_COUNT )
uint32_t L1PParityL1PData[L1P_PARITY_XFER_COUNT/4U];

/* L1P_PARITY_XFER_COUNT bytes aligned to 4byte boundary reserved in L2 for L1P Parity diagnostics */
#pragma DATA_SECTION (L1PParityL2Data, "l2_diag_data");
#pragma DATA_ALIGN ( L1PParityL2Data , 4 )
uint32_t L1PParityL2Data[L1P_PARITY_XFER_COUNT/4U];


/* Reserve memory in L3 memory for L3 ECC error injection */
#pragma DATA_SECTION (L3ECCData, "l3ecc_diag_data");
#pragma DATA_ALIGN ( L3ECCData , L3_ECC_RESERVED_COUNT )
uint32_t L3ECCData[L3_ECC_RESERVED_COUNT/4U];


/* Reserve memory in L2 UMAP0 for L2 parity error injection */
#pragma DATA_SECTION (L2ParityDataUmap0, "l2parity_diag_data_umap0");
#pragma DATA_ALIGN ( L2ParityDataUmap0 , L2_PARITY_RESERVED_COUNT )
uint32_t L2ParityDataUmap0[L2_PARITY_RESERVED_COUNT/4U];

/* Reserve memory in L2 UMAP1 for L2 parity error injection */
#pragma DATA_SECTION (L2ParityDataUmap1, "l2parity_diag_data_umap1");
#pragma DATA_ALIGN ( L2ParityDataUmap1 , L2_PARITY_RESERVED_COUNT )
uint32_t L2ParityDataUmap1[L2_PARITY_RESERVED_COUNT/4U];

/**
 * @brief   IDMA1 Hook counter which counts the number of times the IDMA 1
 * interrupt hook was invoked.
 */
volatile uint8_t        gIDMA1InterruptHookCounter;

/* DSS Diagnostic Test Status Bit
 * This stores the Diagnostic test status at each bits */
uint32_t gDssDiagTestStatus = 0;
/* Each bit stands for executed Diag Test on DSS */
uint32_t gDssDiagTestExec   = 0;

extern void OSAL_C674_ESMDrv_init (void);
extern void OSAL_C674_CycleProfiler_init (void);

/**
 *  @b Description
 *  @n
 *      The function is the interrupt handler which is registered by the
 *      test and should be invoked to indicate that the IDMA Channel 1
 *      interrupt occurred.
 *
 *  @retval
 *      Not applicable.
 */
static void Test_IDMA1_notifyInterruptHandler(void)
{
    /* Interrupt Handler has been invoked. */
    gIDMA1InterruptHookCounter = gIDMA1InterruptHookCounter + 1U;
}


/**
 *  @b Description
 *  @n
 *      The function is used for polling IDMA1 completion.
 *
 *  @retval
 *      1 - IDMA1 interrupt handler was invoked
 *  @retval
 *      0 - IDMA1 interrupt handler was not invoked
 */
static uint32_t Test_L2_ECC_isIDMA1NotifyInvoked (void)
{
    uint32_t    done = 0U;

    /* IDMA interrupt invoked? */
    if (gIDMA1InterruptHookCounter > 0U)
    {
        /* YES. */
        done = 1U;
        gIDMA1InterruptHookCounter = 0U;
    }
    return done;
}


/**
 *  @b Description
 *  @n
 *      The function does IDMA1 transfer with the specified parameters.
 *
 *  @param[in] src
 *      Source address to IDMA from
 *  @param[in] dest
 *      Destination address to IDMA into
 *  @param[in] count
 *      Number of bytes to IDMA
 *
 *  @retval
 *      Not applicable.
 */
static void Test_L2_ECC_IDMA1XferUtil(uint32_t src, uint32_t dest, uint32_t count)
{
    CSL_DSPICFGRegs*          ptrDSPICFGRegs;
    CSL_DSPICFG_IDMA1_Config  IDMA1Cfg;
    /* IDMA Channel 1 interrupt number. */
    uint8_t                   IDMA1Interrupt;

    /* Get the base address of DSP_ICFG module*/
    ptrDSPICFGRegs = CSL_DSPICFG_getBaseAddress();

    /* Get the IDMA Channel 1 interrupt number */
    IDMA1Interrupt = Diag_C674_getIDMA1InterruptEvent();

    /* Add the interrupt handler for IDMA Channel 1 interrupt that
     * will be triggered when transfer is done.*/
    OSAL_C674_Interrupt_addHook(IDMA1Interrupt, Test_IDMA1_notifyInterruptHandler);

    while (count > 0U)
    {
        /* Set the IDMA src address */
        CSL_DSPICFG_IDMA1_setSrcAddr(ptrDSPICFGRegs, src);

        /* Set the IDMA dest address */
        CSL_DSPICFG_IDMA1_setDestAddr(ptrDSPICFGRegs, dest);

        /* Set the IDMA count, no fill, interrupt enable, low priority */
        IDMA1Cfg.count = (count > L2_ECC_IDMA_SINGLE_XFER_COUNT) ? L2_ECC_IDMA_SINGLE_XFER_COUNT : count;
        IDMA1Cfg.fill = 0U;
        IDMA1Cfg.intEnable = 1U;
        IDMA1Cfg.priority = 7U;
        CSL_DSPICFG_IDMA1_setCountAndCfg(ptrDSPICFGRegs, &IDMA1Cfg);
        if (count > L2_ECC_IDMA_SINGLE_XFER_COUNT)
        {
            count = count - L2_ECC_IDMA_SINGLE_XFER_COUNT;
            src = src + L2_ECC_IDMA_SINGLE_XFER_COUNT;
            dest = dest + L2_ECC_IDMA_SINGLE_XFER_COUNT;
        }
        else
        {
            count = 0U;
        }

        /* Wait for IDMA Channel 1 interrupt to happen */
        (void)Diag_poll(100000U, Test_L2_ECC_isIDMA1NotifyInvoked);
    }

    /* Delete the IDMA1 Interrupt Hook. */
    OSAL_C674_Interrupt_delHook (IDMA1Interrupt);

    /* If the IDMA transfer times out the test should fail. No additional check is needed */
}

/**
 *  @b Description
 *  @n
 *      This is the implementation of the HWA lockstep diagnostic.
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
int32_t DssDiag_HwaLockstepTest (uint32_t* ptrCycleCount)
{
    int32_t               retVal;

    /********************************************************************
     * Inject HWA lockstep error
     ********************************************************************/
    /* Execute the diagnostic: */
    retVal = Diag_C674_HWA_lockstep_execute(HWA_LOCKSTEP_MAX_CYCLE_COUNT, ptrCycleCount);
    if(retVal == 0)
    {
        printf ("[PASS] Diag C674 HWA Lockstep test\n");
    }
    else
    {
        printf ("[FAIL] Diag C674 HWA Lockstep test %d\n", retVal);
    }
    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_HWA_LOCKSTEP_TEST_STATUS_BIT);
    DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_HWA_LOCKSTEP_TEST_STATUS_BIT);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      The function is used to test the HWA ECC Diagnostic
 *      with basic configuration. The results of the test execution
 *      are logged into the framework
 *
 *  @retval
 *      Not applicable
 */
int32_t DssDiag_HwaEccTest (Diag_ECCMode eccMode, uint32_t* ptrCycleCount)
{
    CSL_DSSRegs*       ptrDSSRegs;
    CSL_HWARegs*       ptrHWARegs;

    Diag_HWA_ECC_Cfg      cfg;
    int32_t               retVal;
    uint32_t              cycleCount = 0;
    CSL_HWA_MemoryType    memType;
    CSL_HWA_MemoryType    startMemType;
    CSL_HWA_MemoryType    endMemType;

    /* Get the base address of DSS_REG Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Get the base address of HWA Module */
    ptrHWARegs = CSL_HWA_getBaseAddress();

    /* Enable DSS HWA (without this step the HWA registers cannot be read) */
    CSL_DSS_setHWAEnableControl (ptrDSSRegs, 1U);

    /* Enable HWA CLK */
    CSL_HWA_setClockEnableControl (ptrHWARegs, 1U);

    startMemType = CSL_HWA_MemoryType_IPING;
#ifdef SOC_XWR68XX
    endMemType = CSL_HWA_MemoryType_MC_PONG;
#else
    endMemType = CSL_HWA_MemoryType_PARAM_RAM;
#endif

    /* Enable ECC and Initialize all memories */
    for (memType = startMemType; memType <= endMemType; memType++)
    {
        /* Enable ECC for memType */
        CSL_HWA_setECCControl(ptrHWARegs, memType, 1U);
        /* Enable memory init for memType */
        CSL_HWA_setMemInit(ptrHWARegs, memType);

        /* Wait for the HWA memory initialization to complete */
        while ((CSL_HWA_getMemInitStatus(ptrHWARegs, memType)) != 1U)
        {
            /* Do nothing */
        }

        /********************************************************************
         * Inject HWA ECC single/double bit error
         ********************************************************************/
        (void)memset ((void *)&cfg, 0, sizeof(Diag_HWA_ECC_Cfg));

        /* Populate the diagnostic configuration: */
        cfg.memType         = memType;
        cfg.eccMode         = eccMode;
        cfg.maxCycleCount   = HWA_ECC_MAX_CYCLE_COUNT;

        /* Execute the diagnostic: */
        retVal = Diag_C674_HWA_ECC_execute(&cfg, ptrCycleCount);
        if(retVal != DIAG_SUCCESS)
        {
            printf ("[ERROR] Diag C674 HWA ECC %d Bit Error [%d]\n", \
                    eccMode, retVal);
            goto EXIT;
        }
        /* update the cyclecount if requested by the func caller */
        UPDATE_DIAG_TMP_TEST_CYCLE_COUNT(cycleCount, ptrCycleCount);
    }

    /* finally setting cycleCount to returned value */
    UPDATE_DIAG_TEST_CYCLE_COUNT(ptrCycleCount, cycleCount);

    printf ("[PASS] Diag C674 HWA ECC %d Bit Error\n", eccMode);
EXIT:
    if(eccMode == Diag_ECCMode_SINGLE_BIT_ERROR)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_HWA_ECC_1B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_HWA_ECC_1B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_HWA_ECC_2B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_HWA_ECC_2B_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      L3 ECC single/double bit error Injection diagnostics
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_L3EccTest(Diag_L3_ECC_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSS2Regs*       ptrDSS2Regs;
    int32_t         retVal;
#if defined (SOC_XWR18XX) || defined (SOC_XWR16XX)
    CSL_TopRCMRegs*     ptrTopRCMRegs;
    /* Get the Top-level RCM register layer base address */
    ptrTopRCMRegs = CSL_TopRCM_getBaseAddress();
#endif
    /* Get the base address of DSS_REG2 Module */
    ptrDSS2Regs = CSL_DSS2_getBaseAddress ();

    /* Enable L3 ECC */
    CSL_DSS_setL3ECCControl (ptrDSS2Regs, 1U);

#if defined (SOC_XWR16XX)
    /* Clear L3 memory and initialize ECC (don't touch reserved banks 0,7) */
    ptrTopRCMRegs->MEMINITSTARTSHMEM = 0x0000007EU;
    while ((ptrTopRCMRegs->MEMINITDONESHMEM & 0x7EU) != 0x7EU);
#elif defined (SOC_XWR18XX)
    /* Clear L3 memory and initialize ECC */
    ptrTopRCMRegs->MEMINITSTARTSHMEM = 0x000000FFU;
    while ((ptrTopRCMRegs->MEMINITDONESHMEM & 0xFFU) != 0xFFU);
#endif

    /********************************************************************
     * Inject L3 ECC single/double bit error
     * (Test with default DSSMEMTAB setting)
     ********************************************************************/
    retVal = Diag_C674_L3_ECC_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag L3 ECC [Inject Single Bit Error at addr 0x%x] [%d]\n", cfg.l3Addr, retVal);
        goto EXIT;
    }
EXIT:
    if(cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L3_ECC_1B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L3_ECC_1B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L3_ECC_2B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L3_ECC_2B_TEST_STATUS_BIT);
    }

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      L2 Parity error Injection diagnostics
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_l2ParityTest(Diag_L2_Parity_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSPICFGRegs*    ptrDSPICFGRegs;
    uint8_t             L1PMode;
    uint8_t             L1DMode;
    uint8_t             L2Mode;
    int32_t             retVal;

    /* Get the base address of DSP_ICFG module*/
    ptrDSPICFGRegs = CSL_DSPICFG_getBaseAddress();

    /* Store current L2 cfg */
    L1PMode = CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs);
    /* Disable L1P cache */
    CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /* Store current L1D cfg */
    L1DMode = CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs);

    /* Disable L1D cache */
    CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /* Store current L2 cache size */
    L2Mode = CSL_DSPICFG_L2Cfg_getMode(ptrDSPICFGRegs);

    /* Disable L2 cache */
    CSL_DSPICFG_L2Cfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L2Cfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /********************************************************************
     * - Inject L2 UMAP0/UMAP1 Parity error
     ********************************************************************/
    retVal = Diag_L2_Parity_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag L2 Parity [Inject UMAP0/1 Parity Error] [%d]\n", retVal);
        goto EXIT;
    }

    /* Note that additional steps maybe needed to restore cache settings */
    /* Restore L2 cfg */
    CSL_DSPICFG_L2Cfg_setMode(ptrDSPICFGRegs, L2Mode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L2Cfg_getMode(ptrDSPICFGRegs) != L2Mode)
    {
    }

    /* Restore L1P cfg */
    CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, L1PMode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs) != L1PMode)
    {
    }

    /* Restore L1D cfg */
    CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, L1DMode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs) != L1DMode)
    {
    }

    printf("[PASS] Diag L2 Parity Error Injection UMAP0/1.\n");
EXIT:
    /* if it is L2SRAM_UMAP0 address range */
    if(cfg.l2Addr >=  0x00800000)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L2P_PARITY_P0_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L2P_PARITY_P0_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L2P_PARITY_P1_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L2P_PARITY_P1_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      L2 ECC error Injection diagnostics
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_l2EccTest(Diag_L2_ECC_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSPICFGRegs*    ptrDSPICFGRegs;
    uint32_t            l2AddrStart;
    uint32_t            l2Size;
    uint8_t             L1PMode;
    uint8_t             L1DMode;
    uint8_t             L2Mode;
    int32_t             retVal = DIAG_SUCCESS;

    /* Get the base address of DSP_ICFG module*/
    ptrDSPICFGRegs = CSL_DSPICFG_getBaseAddress();

    /* Store current L1P cfg */
    L1PMode = CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs);
    /* Disable L1P cache */
    CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /* Store current L1D cfg */
    L1DMode = CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs);

    /* Disable L1D cache */
    CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /* Store current L2 cache size */
    L2Mode = CSL_DSPICFG_L2Cfg_getMode(ptrDSPICFGRegs);

    /* Disable L2 cache */
    CSL_DSPICFG_L2Cfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L2Cfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /* Get L2 base address and size */
    l2AddrStart = CSL_DSS_getUMAPBaseAddr(1U);
    l2Size = CSL_DSS_getL2Size();

    /* Enables L2 ECC for all pages of UMAP0 */
    CSL_DSPICFG_L2ECC_setPageEnable(ptrDSPICFGRegs, 0U, 0xFFFFFFFFU);
    /* Enables L2 ECC for all pages of UMAP1 */
    CSL_DSPICFG_L2ECC_setPageEnable(ptrDSPICFGRegs, 1U, 0xFFFFFFFFU);

    /* Enable ECC in L2 and initialize ECC */
    Diag_L2_ECC_setEDCMode(ptrDSPICFGRegs, CSL_EDC_ENABLE);

    /* Run a IDMA transfer across complete L2 memory to initialize ECC memory */
    Test_L2_ECC_IDMA1XferUtil(l2AddrStart, l2AddrStart, l2Size);

    /********************************************************************
     * - Inject L2 ECC single/Double bit error
     ********************************************************************/
    retVal = Diag_L2_ECC_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag L2 ECC [%d] bit error failed with [%d]\n", cfg.eccMode, retVal);
        goto EXIT;
    }

    /* Note that additional steps maybe needed to restore cache settings */
    /* Restore L2 cfg */
    CSL_DSPICFG_L2Cfg_setMode(ptrDSPICFGRegs, L2Mode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L2Cfg_getMode(ptrDSPICFGRegs) != L2Mode)
    {
    }

    /* Restore L1P cfg */
    CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, L1PMode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs) != L1PMode)
    {
    }

    /* Restore L1D cfg */
    CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, L1DMode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs) != L1DMode)
    {
    }

    printf("[PASS]  Diag L2 ECC [%d] bit error injection.\n", cfg.eccMode);
EXIT:
    if(cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L2_ECC_1B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L2_ECC_1B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L2_ECC_2B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L2_ECC_2B_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      L1P Parity Error Injection diagnostic
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_l1pParityTest(Diag_L1P_Parity_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSPICFGRegs*    ptrDSPICFGRegs;
    uint8_t             L1PMode;
    uint8_t             L1DMode;
    int32_t             retVal;

    /* Get the base address of DSP_ICFG module*/
    ptrDSPICFGRegs = CSL_DSPICFG_getBaseAddress();

    /* Store current L1P cfg */
    L1PMode = CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs);
    /* Disable L1P cache */
    CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /* Store current L1D cfg */
    L1DMode = CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs);

    /* Disable L1D cache */
    CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, 0U);
    /* Wait for the cache disable status to reflect */
    while (CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs) != 0U)
    {
    }

    /********************************************************************
     * - Execute the diagnostic
     ********************************************************************/
    retVal = Diag_L1P_Parity_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag L1P Parity [Insert Parity Error] [%d]\n", retVal);
        //goto EXIT; //Don't EXIT without restoring the L1P cfg
    }
    else
    {
        printf("[PASS] Diag L1P Parity.\n");
    }

    /* Restore L1P cfg */
    CSL_DSPICFG_L1PCfg_setMode(ptrDSPICFGRegs, L1PMode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L1PCfg_getMode(ptrDSPICFGRegs) != L1PMode)
    {
    }

    /* Restore L1D cfg */
    CSL_DSPICFG_L1DCfg_setMode(ptrDSPICFGRegs, L1DMode);
    /* Wait for the cfg status to reflect */
    while (CSL_DSPICFG_L1DCfg_getMode(ptrDSPICFGRegs) != L1DMode)
    {
    }

    /* Set Diagnostic Test Status to global status field */
    DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_L1P_PARITY_TEST_STATUS_BIT);
    /* Set Text Execution bit to indicate that this test has executed */
    DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_L1P_PARITY_TEST_STATUS_BIT);
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      HSRAM ECC Error Injection diagnostic
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_hsramEccTest(Diag_HSRAM_ECC_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSSRegs*       ptrDSSRegs;
    int32_t            retVal;

    /* Get the base address of DSS_REG Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Enable HSRAM ECC */
    CSL_DSS_setHSRAMECCControl(ptrDSSRegs, 1U);

    /* Initialize HSRAM ECC memory */
    CSL_DSS_setHSRAMECCInit(ptrDSSRegs);
    /* Wait for the HSRAM ECC memory init to complete */
    while ((CSL_DSS_getHSRAMECCInitStatus(ptrDSSRegs)) != 1U);

    /********************************************************************
     * - Inject HSRAM ECC single/double bit error
     ********************************************************************/
    retVal = Diag_C674_HSRAM_ECC_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag HSRAM ECC [Inject Single Bit Error at addr 0x%x] [%d]\n", cfg.HSRAMAddr, retVal);
        goto EXIT;
    }

    printf("[PASS] Diag HSRAM ECC [Inject %d Bit Error at addr 0x%x]\n", cfg.eccMode, cfg.HSRAMAddr);
EXIT:
    if(cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_HSRAM_ECC_1B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_HSRAM_ECC_1B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_HSRAM_ECC_2B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_HSRAM_ECC_2B_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      EDMA TPCC Parity error Injection diagnostic
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_edmaParityTest(Diag_EDMA_Parity_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSSRegs*    ptrDSSRegs;
    uint8_t         done;
    int32_t         retVal;

    /* Get the base address of the CSL Modules: */
    ptrDSSRegs = CSL_DSS_getBaseAddress();

    /* Initialize the TPCC [0 and 1] Parity Memory: */
    CSL_DSS_initTPCCMemory (ptrDSSRegs, cfg.tpccId);

    /* Loop around till the memory initialization is complete:
     * This is done for the TPCC-1 */
    do
    {
        done = CSL_DSS_isTPCCMemoryInitialized (ptrDSSRegs, cfg.tpccId);
    } while (done == 0U);


    /********************************************************************
     * EDMA Instance 0/1 based on input:
     * Initialize the configuration:
     ********************************************************************/
    retVal = Diag_EDMA_Parity_execute (&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag EDMA Parity [TPCC Instance %d] [%d]\n", cfg.tpccId, retVal);
        goto EXIT;
    }
    printf("[PASS] Diag EDMA Parity [TPCC Instance %d]\n", cfg.tpccId);
EXIT:
    if(cfg.tpccId == 0)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_EDMA_PARITY_C0_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_EDMA_PARITY_C0_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_EDMA_PARITY_C1_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_EDMA_PARITY_C1_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      RAM ECC error Injection diagnostics
 *
 *  @param[in]  cfg - diag configuration
 *  @param[out]  ptrCycleCount
 *      [Optional argument] This is used to record the time taken to
 *      execute the diagnostic for benchmarking information.
 *
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_txfrRamEccTest(Diag_DataTxfrRAM_ECC_Cfg cfg, uint32_t* ptrCycleCount)
{
    CSL_DSSRegs*       ptrDSSRegs;
    int32_t            retVal;

    /* Get the base address of DSS_REG Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Enable DataTxfrRAM ECC */
    CSL_DSS_setDataTxfrRAMECCControl(ptrDSSRegs, 1U);

    /* Initialize DataTxfrRAM ECC memory */
    CSL_DSS_setDataTxfrRAMECCInit(ptrDSSRegs);
    /* Wait for the DataTxfrRAM ECC memory init to complete */
    while ((CSL_DSS_getDataTxfrRAMECCInitStatus(ptrDSSRegs)) != 1U);

    /********************************************************************
     * - Inject DataTxfrRAM ECC single bit error
     ********************************************************************/
    retVal = Diag_C674_DataTxfrRAM_ECC_execute(&cfg, ptrCycleCount);
    if (retVal != DIAG_SUCCESS)
    {
        /* Test Result Reporting. */
        printf("[ERROR] Diag DataTxfrRAM ECC [Inject Single/Double Bit Error at addr 0x%x] [%d]\n", \
               cfg.dataTxfrRAMAddr, retVal);
        goto EXIT;
    }
    printf("Diag DataTxfrRAM ECC [Inject %d Bit Error at addr 0x%x]\n", cfg.eccMode, cfg.dataTxfrRAMAddr);
EXIT:
    if(cfg.eccMode == Diag_ECCMode_SINGLE_BIT_ERROR)
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_TXFR_RAM_ECC_1B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_TXFR_RAM_ECC_1B_TEST_STATUS_BIT);
    }
    else
    {
        /* Set Diagnostic Test Status to global status field */
        DIAG_TEST_STATUS_SET(retVal, gDssDiagTestStatus, DIAG_DSS_TXFR_RAM_ECC_2B_TEST_STATUS_BIT);
        /* Set Text Execution bit to indicate that this test has executed */
        DIAG_TEST_EXECUTE_SET(gDssDiagTestExec, DIAG_DSS_TXFR_RAM_ECC_2B_TEST_STATUS_BIT);
    }
    return retVal;
}

/**
 *  @b Description
 *  @n
 *      Execute all the Error Injection Test using SDL on DSS.
 *
 *  @param[in]  None.
 *  @retval
 *      Success -   DIAG_SUCCESS
 *  @retval
 *      Error   -   Diagnostic Error code (\ref DIAG_ERROR_CODE)
 */
int32_t DssDiag_InjectTest(void)
{
    uint32_t cycleCount = 0;
    Diag_DataTxfrRAM_ECC_Cfg    ramEcccfg = {0};
    Diag_EDMA_Parity_Cfg        edmaParityCfg = {0};
    Diag_HSRAM_ECC_Cfg          hsramEccCfg = {0};
    Diag_L1P_Parity_Cfg         l1pParityCfg = {0};
    Diag_L2_ECC_Cfg             l2EccCfg;
    Diag_L2_Parity_Cfg          l2ParityCfg = {0};
    Diag_L3_ECC_Cfg             l3EccCfg = {0};
    int32_t   retVal;

    /** EDMA TPCC Parity Error injection Test */
    /* Populate the diagnostic configuration: */
    edmaParityCfg.tpccId        = 0;
    edmaParityCfg.paramSet      = TPCC0_PARAMSET_SELECTION;
    edmaParityCfg.maxCycleCount = 1000U;
    retVal = DssDiag_edmaParityTest(edmaParityCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** EDMA TPCC1 Parity Error injection Test */
    /* Populate the diagnostic configuration: */
    edmaParityCfg.tpccId        = 1;
    edmaParityCfg.paramSet      = TPCC1_PARAMSET_SELECTION;
    edmaParityCfg.maxCycleCount = 1000U;
    retVal = DssDiag_edmaParityTest(edmaParityCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** Inject L2 UMAP0/UMAP1 Parity error Test */
    /* Populate the diagnostic configuration: */
    /* L2 UMAP0 memory location.
     * L2 address should be 4 byte aligned */
    l2ParityCfg.l2Addr = (uint32_t)&L2ParityDataUmap0;
    l2ParityCfg.maxCycleCount = L2_PARITY_MAX_CYCLE_COUNT;
    retVal = DssDiag_l2ParityTest(l2ParityCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** Inject L2 UMAP0/UMAP1 Parity error Test */
    /* Populate the diagnostic configuration: */
    /* L2 UMAP1 memory location.
     * L2 address should be 4 byte aligned */
    l2ParityCfg.l2Addr = (uint32_t)&L2ParityDataUmap1;
    l2ParityCfg.maxCycleCount = L2_PARITY_MAX_CYCLE_COUNT;
    retVal = DssDiag_l2ParityTest(l2ParityCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** L1P Parity Error injection Test */
    /* Populate the diagnostic configuration: */
    l1pParityCfg.l1pAddr       = (uint32_t)L1PParityL1PData; /* L1P address should be 32 byte aligned */
    l1pParityCfg.l2Addr        = (uint32_t)L1PParityL2Data;  /* L2 address should be 4 byte aligned */
    l1pParityCfg.count         = L1P_PARITY_XFER_COUNT;
    l1pParityCfg.maxCycleCount = L1P_PARITY_MAX_CYCLE_COUNT;
    retVal = DssDiag_l1pParityTest(l1pParityCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** L2 ECC Single Bit Error injection Test */
    /* Populate the diagnostic configuration: */
    l2EccCfg.eccMode       = Diag_ECCMode_SINGLE_BIT_ERROR;
    l2EccCfg.maxCycleCount = L2_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_l2EccTest(l2EccCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** L2 ECC Double Bit Error injection Test */
    /* Populate the diagnostic configuration: */
    l2EccCfg.eccMode       = Diag_ECCMode_DOUBLE_BIT_ERROR;
    l2EccCfg.maxCycleCount = L2_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_l2EccTest(l2EccCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** Inject L3 ECC single Bit error Test */
    /* Populate the diagnostic configuration: */
    l3EccCfg.l3Addr        = (uint32_t)&L3ECCData;
    l3EccCfg.eccMode       = Diag_ECCMode_SINGLE_BIT_ERROR;
    l3EccCfg.maxCycleCount = L3_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_L3EccTest(l3EccCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** Inject L3 ECC Double Bit error Test */
    /* Populate the diagnostic configuration: */
    l3EccCfg.l3Addr        = (uint32_t)&L3ECCData;
    l3EccCfg.eccMode       = Diag_ECCMode_DOUBLE_BIT_ERROR;
    l3EccCfg.maxCycleCount = L3_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_L3EccTest(l3EccCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** HSRAM ECC single Bit Error injection Test */
    /* Populate the diagnostic configuration: */
    hsramEccCfg.eccMode = Diag_ECCMode_SINGLE_BIT_ERROR;
    hsramEccCfg.HSRAMAddr = (uint32_t)&HSRAMECCData[0];
    hsramEccCfg.maxCycleCount   = HSRAM_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_hsramEccTest(hsramEccCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** HSRAM ECC Double Bit Error injection Test */
    /* Populate the diagnostic configuration: */
    hsramEccCfg.eccMode = Diag_ECCMode_DOUBLE_BIT_ERROR;
    hsramEccCfg.HSRAMAddr = (uint32_t)&HSRAMECCData[3];
    hsramEccCfg.maxCycleCount   = HSRAM_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_hsramEccTest(hsramEccCfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /* HWA Memory Single Bit ECC Error Injection Diagnostic Test */
    retVal = DssDiag_HwaEccTest(Diag_ECCMode_SINGLE_BIT_ERROR, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /* HWA Memory Double Bit ECC Error Injection Diagnostic Test */
    retVal = DssDiag_HwaEccTest(Diag_ECCMode_DOUBLE_BIT_ERROR, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /* HWA Lock Step Error Injection Diagnostic Test */
    retVal = DssDiag_HwaLockstepTest(&cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** RAM Single Bit ECC error Injection diagnostics  */
    /* Populate the diagnostic configuration: */
    ramEcccfg.eccMode         = Diag_ECCMode_SINGLE_BIT_ERROR;
    ramEcccfg.dataTxfrRAMAddr = (uint32_t)&DataTxfrRAMECCData[0];
    ramEcccfg.maxCycleCount   = DATA_TXFR_RAM_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_txfrRamEccTest(ramEcccfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    /** RAM Double Bit ECC error Injection diagnostics  */
    /* Populate the diagnostic configuration: */
    ramEcccfg.eccMode         = Diag_ECCMode_DOUBLE_BIT_ERROR;
    ramEcccfg.dataTxfrRAMAddr = (uint32_t)&DataTxfrRAMECCData[3];
    ramEcccfg.maxCycleCount   = DATA_TXFR_RAM_ECC_MAX_CYCLE_COUNT;
    retVal = DssDiag_txfrRamEccTest(ramEcccfg, &cycleCount);
    RETURN_IF_ERROR(retVal == 0);

    return retVal;
}

/**
 *  @b Description
 *  @n
 *      Function to initialize the C674x Interrupt, ESM and Cycle profiler
 *
 *  @retval  None
 */
void DssDiag_IntEsmDrvInit()
{
    /* Interrupt INIT, event combiner and global interrupt enable
     * happen within soc_init call */
    //OSAL_C674_Interrupt_init();

    /* Initialize the OSAL Cycle Profiling Module: */
    OSAL_C674_CycleProfiler_init ();

    /* Initialize the OSAL ESM Driver: */
    OSAL_C674_ESMDrv_init ();
}


