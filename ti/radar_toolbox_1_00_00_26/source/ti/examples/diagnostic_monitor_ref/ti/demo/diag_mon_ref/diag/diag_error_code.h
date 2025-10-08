/**
 *   @file  diag_error_code.h
 *
 *   @brief
 *     This file contains all error codes for SDL tests status for application.
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


#ifndef DIAG_ERROR_CODE_H
#define DIAG_ERROR_CODE_H

#ifdef __cplusplus
extern "C" {
#endif

#define DIAG_TEST_STATUS_SET(a, b, c)     if(a!=0){b|=(((uint64_t)1)<<c);}
#define DIAG_TEST_EXECUTE_SET(a, b)       a = (a |(1<<b));

/**** Error Code On MSS SDL Test Execution ****/
/* Boot time test error from Bootloader */
#define DIAG_BOOT_DCC_FAULT_INSR_TEST_ERROR               -100
#define DIAG_BOOT_DCC_FUNC_TEST_ERROR                     -101
#define DIAG_BOOT_DMA_FUNC_TEST_ERROR                     -102
#define DIAG_BOOT_DMA_MPU_TEST_ERROR                      -103
#define DIAG_BOOT_DMA_RAM_PARITY_TEST_ERROR               -104
#define DIAG_BOOT_ESM_FUNC_TEST_ERROR                     -105
#define DIAG_BOOT_ESM_GRP1_ERROR                          -106
#define DIAG_BOOT_MEM_INIT_TEST_ERROR                     -107
#define DIAG_BOOT_MIBSPI_RAM_DOUBLE_BIT_ERROR             -108
#define DIAG_BOOT_MIBSPI_RAM_SINGLE_BIT_ERROR             -109
#define DIAG_BOOT_MIBSPI_TEST_ERROR                       -110
#define DIAG_BOOT_MPU_FUNC_TEST_ERROR                     -111
#define DIAG_BOOT_MSS_CRC_TEST_ERROR                      -112
#define DIAG_BOOT_PBIST_DUAL_PORT_MEM_TEST_ERROR          -113
#define DIAG_BOOT_PBIST_SINGLE_PORT_MEM_TEST_ERROR        -114
#define DIAG_BOOT_PCR_FAULT_INSR_TEST_ERROR               -115
#define DIAG_BOOT_ROM_CRC_TEST_ERROR                      -116
#define DIAG_BOOT_RTI_FUNC_TEST_ERROR                     -117
#define DIAG_BOOT_UART_FUNC_TEST_ERROR                    -118
#define DIAG_BOOT_VIM_FUNC_TEST_ERROR                     -119
#define DIAG_BOOT_VIM_RAM_PARITY_TEST_ERROR               -120
#define DIAG_BOOT_MSS_LBIST_TEST_ERROR                    -121
#define DIAG_BOOT_DSS_LBIST_PBIST_TEST_ERROR              -122
#define DIAG_BOOT_RESERVED0__ERROR                        -123
#define DIAG_BOOT_RESERVED1__ERROR                        -124
#define DIAG_BOOT_RESERVED2__ERROR                        -125

/* Peripheral or Memory error when SDL Diagnostic test executes */
#define DIAG_VIM_ECC_TEST_ERROR                           -126
#define DIAG_VIM_STATIC_TEST_ERROR                        -127


/**********************************************************
 * Bit mapping to type of Diagnostic Test on MSS.
 * 1: Test Failed, 0: Test Passed
 **********************************************************/
/* Error Injection Error Diagnostic Test Status bit */
#define DIAG_MSS_BOOT_TEST_STATUS_BIT                   0
#define DIAG_MSS_TCM_A_PARITY_TEST_STATUS_BIT           1
#define DIAG_MSS_TCM_B0_PARITY_TEST_STATUS_BIT          2
#define DIAG_MSS_TCM_B1_PARITY_TEST_STATUS_BIT          3
#define DIAG_MSS_TCM_A_ECC_1B_TEST_STATUS_BIT           4
#define DIAG_MSS_TCM_A_ECC_2B_TEST_STATUS_BIT           5
#define DIAG_MSS_TCM_B0_ECC_1B_TEST_STATUS_BIT          6
#define DIAG_MSS_TCM_B0_ECC_2B_TEST_STATUS_BIT          7
#define DIAG_MSS_TCM_B1_ECC_1B_TEST_STATUS_BIT          8
#define DIAG_MSS_TCM_B1_ECC_2B_TEST_STATUS_BIT          9
#define DIAG_MSS_DMA0_MPU_TEST_STATUS_BIT               10
#define DIAG_MSS_DMA1_MPU_TEST_STATUS_BIT               11
#define DIAG_MSS_DMA0_PARITY_TEST_STATUS_BIT            12
#define DIAG_MSS_DMA1_PARITY_TEST_STATUS_BIT            13
#define DIAG_MSS_WATCHDOG_TEST_STATUS_BIT               14
#define DIAG_MSS_VIM_ECC_1B_TEST_STATUS_BIT             15
#define DIAG_MSS_VIM_ECC_2B_TEST_STATUS_BIT             16
#define DIAG_MSS_MIBSPI0_ECC_1B_TEST_STATUS_BIT         17
#define DIAG_MSS_MIBSPI0_ECC_2B_TEST_STATUS_BIT         18
#define DIAG_MSS_MIBSPI1_ECC_1B_TEST_STATUS_BIT         19
#define DIAG_MSS_MIBSPI1_ECC_2B_TEST_STATUS_BIT         20
#define DIAG_MSS_MAILBOX_ECC_1B_TEST_STATUS_BIT         21
#define DIAG_MSS_MAILBOX_ECC_2B_TEST_STATUS_BIT         22
#define DIAG_MSS_HWA_ECC_1B_TEST_STATUS_BIT             23
#define DIAG_MSS_HWA_ECC_2B_TEST_STATUS_BIT             24
#define DIAG_MSS_L3_ECC_1B_TEST_STATUS_BIT              25
#define DIAG_MSS_L3_ECC_2B_TEST_STATUS_BIT              26
#define DIAG_MSS_TXFR_RAM_ECC_1B_TEST_STATUS_BIT        27
#define DIAG_MSS_TXFR_RAM_ECC_2B_TEST_STATUS_BIT        28
#define DIAG_MSS_HSRAM_ECC_1B_TEST_STATUS_BIT           29
#define DIAG_MSS_HSRAM_ECC_2B_TEST_STATUS_BIT           30
/* SELF TEST Diagnostic Test Status Bit
 * STC and PBIST are ran by SBL (secondary Bootloader) */
#define DIAG_MSS_PBIST_TEST_STATUS_BIT                  31
#define DIAG_MSS_STC_TEST_STATUS_BIT                    32
#define DIAG_MSS_DCCA_TEST_STATUS_BIT                   33
#define DIAG_MSS_DCCB_TEST_STATUS_BIT                   34
#define DIAG_MSS_CCMA_TEST_STATUS_BIT                   35
#define DIAG_MSS_CCMB_TEST_STATUS_BIT                   36
#define DIAG_MSS_HWA_LOCKSTEP_TEST_STATUS_BIT           37
/* LOOPBACK Peripheral Diagnostic Test Status Bit */
#define DIAG_MSS_MCAN_LOOPBACK_TEST_STATUS_BIT          38
#define DIAG_MSS_UART0_LOOPBACK_TEST_STATUS_BIT         39
#define DIAG_MSS_UART1_LOOPBACK_TEST_STATUS_BIT         40
#define DIAG_MSS_MIBSPI0_LOOPBACK_TEST_STATUS_BIT       41
#define DIAG_MSS_MIBSPI1_LOOPBACK_TEST_STATUS_BIT       42
#define DIAG_MSS_I2C_LOOPBACK_TEST_STATUS_BIT           43
#define DIAG_MSS_DCAN_LOOPBACK_TEST_STATUS_BIT          44
#define DIAG_MSS_NERROR_IN_TEST_STATUS_BIT              45
#define DIAG_MSS_NERROR_OUT_TEST_STATUS_BIT             46
/* Readback of STATIC CONFIG Diagnostic Test Status Bit */
#define DIAG_MSS_DMA0_STATIC_TEST_STATUS_BIT            47
#define DIAG_MSS_DMA1_STATIC_TEST_STATUS_BIT            48
#define DIAG_MSS_RTI_STATIC_TEST_STATUS_BIT             49
#define DIAG_MSS_VIM_STATIC_TEST_STATUS_BIT             50
#define DIAG_MSS_MIBSPI0_STATIC_TEST_STATUS_BIT         51
#define DIAG_MSS_MIBSPI1_STATIC_TEST_STATUS_BIT         52
#define DIAG_MSS_DCAN_STATIC_TEST_STATUS_BIT            53
#define DIAG_MSS_MCAN_STATIC_TEST_STATUS_BIT            54
#define DIAG_MSS_ESM_STATIC_TEST_STATUS_BIT             55
#define DIAG_MSS_RCM_STATIC_TEST_STATUS_BIT             56
#define DIAG_MSS_R4F_STATIC_TEST_STATUS_BIT             57
#define DIAG_MSS_MAX_TEST_STATUS_BIT                    58


/***** Error Code On DSS SDL Test Execution *****/


/**********************************************************
 * Bit mapping to type of Diagnostic Test on DSS.
 * 1: Test Failed, 0: Test Passed
 **********************************************************/
/* Error Injection Error Diagnostic Test Status bit */
#define DIAG_DSS_EDMA_PARITY_C0_TEST_STATUS_BIT              0
#define DIAG_DSS_EDMA_PARITY_C1_TEST_STATUS_BIT              1
#define DIAG_DSS_L1P_PARITY_TEST_STATUS_BIT                  2
#define DIAG_DSS_L2P_PARITY_P0_TEST_STATUS_BIT               3
#define DIAG_DSS_L2P_PARITY_P1_TEST_STATUS_BIT               4
#define DIAG_DSS_L2_ECC_1B_TEST_STATUS_BIT                   5
#define DIAG_DSS_L2_ECC_2B_TEST_STATUS_BIT                   6
#define DIAG_DSS_L3_ECC_1B_TEST_STATUS_BIT                   7
#define DIAG_DSS_L3_ECC_2B_TEST_STATUS_BIT                   8
#define DIAG_DSS_TXFR_RAM_ECC_1B_TEST_STATUS_BIT             9
#define DIAG_DSS_TXFR_RAM_ECC_2B_TEST_STATUS_BIT             10
#define DIAG_DSS_HSRAM_ECC_1B_TEST_STATUS_BIT                11
#define DIAG_DSS_HSRAM_ECC_2B_TEST_STATUS_BIT                12
#define DIAG_DSS_HWA_ECC_1B_TEST_STATUS_BIT                  13
#define DIAG_DSS_HWA_ECC_2B_TEST_STATUS_BIT                  14
/* Self Test Diagnostic Test Status Bit */
#define DIAG_DSS_HWA_LOCKSTEP_TEST_STATUS_BIT                15
/* STC and PBIST are ran by SBL (secondary Bootloader) */
#define DIAG_DSS_STC_TEST_STATUS_BIT                         16
#define DIAG_DSS_PBIST_TEST_STATUS_BIT                       17

#define DIAG_DSS_MAX_TEST_STATUS_BIT                         18

#ifdef __cplusplus
}
#endif

#endif /* DIAG_ERROR_CODE_H */
