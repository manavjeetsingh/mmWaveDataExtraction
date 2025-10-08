/**
 *   @file  diag_test_api.h
 *
 *   @brief
 *      This is the API declaration of all the Diagnostic Test APIs for MSS and DSS.
 *
 *  \par
 *  NOTE:
<<<<<<< HEAD
 *      (C) Copyright 2020 Texas Instruments, Inc.
=======
 *      (C) Copyright 2016 Texas Instruments, Inc.
>>>>>>> 6369315f8b1ba2de6c8d83d641b8e697843b04e3
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
#ifndef DIAG_TEST_API_H
#define DIAG_TEST_API_H

#include <ti/diag/diag.h>


/* DSS Diagnostic Test Functions */
int32_t DssDiag_HwaLockstepTest (uint32_t* ptrCycleCount);
int32_t DssDiag_HwaEccTest (Diag_ECCMode eccMode, uint32_t* ptrCycleCount);
int32_t DssDiag_L3EccTest(Diag_L3_ECC_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_l2ParityTest(Diag_L2_Parity_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_l2EccTest(Diag_L2_ECC_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_l1pParityTest(Diag_L1P_Parity_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_hsramEccTest(Diag_HSRAM_ECC_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_edmaParityTest(Diag_EDMA_Parity_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_txfrRamEccTest(Diag_DataTxfrRAM_ECC_Cfg cfg, uint32_t* ptrCycleCount);
int32_t DssDiag_InjectTest(void);
void DssDiag_IntEsmDrvInit();

/* MSS Diagnostic Test Functions */
void MssDiag_IntEsmDrvInit();
int32_t MssDiag_ErrorInjectTest(void);
int32_t MssDiag_StaticConfigTest(void);
int32_t MssDiag_UartLoopbackTest(uint8_t instanceId, uint32_t* ptrCycleCount);
int32_t MssDiag_UartLoopbackTest(uint8_t instanceId, uint32_t* ptrCycleCount);
int32_t MssDiag_MibSPILoopbackTest (uint8_t instanceId, uint32_t* ptrCycleCount);
int32_t MssDiag_McanLoopbackTest(uint32_t* ptrCycleCount);
int32_t MssDiag_I2cLoopbackTest(uint32_t* ptrCycleCount);
int32_t MssDiag_DcanLoopbackTest(uint32_t* ptrCycleCount);
int32_t MssDiag_RtiStaticCfgTest (Diag_StaticCfgErrInfo *ptrErrInfo, uint32_t *ptrCycleCount);
int32_t MssDiag_WatchdogTest (uint32_t* ptrCycleCount);
int32_t MssDiag_vimEccTest (Diag_VIM_ECC_Cfg cfg, uint32_t* ptrCycleCount);
int32_t MssDiag_vimStaticTest(Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_vimStaticCfg_verifyViolationWakeup (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_vimStaticCfg_verifyViolationFallbackAddr (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_vimStaticCfg_verifyViolationECCDiag (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_vimStaticCfg_verifyViolationChanCtrl (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_dmaMpuTest (Diag_DMA_MPU_Cfg cfg, uint32_t* ptrCycleCount);
int32_t MssDiag_dmaParityTest (Diag_DMA_Parity_Cfg cfg, uint32_t* ptrCycleCount);
int32_t MssDiag_dmaStaticTest (uint8_t dmaInstanceId, Diag_StaticCfgErrInfo *ptrErrorInfo,uint32_t* ptrCycleCount);
int32_t MssDiag_dmaStaticCfg_verifyMPU (uint8_t dmaInstanceId, Diag_StaticCfgErrInfo* ptrErrorInfo,uint32_t* ptrCycleCount);
int32_t MssDiag_MibspiEccTest (Diag_MIBSPI_ECC_Cfg  cfg,Diag_MIBSPI_ECC_ErrorInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_MibspiStaticTest (uint8_t instanceId,Diag_StaticCfgErrInfo* ptrErrorInfo,uint32_t* ptrCycleCount);
int32_t MssDiag_MibSPiStaticViolationTest (uint8_t instanceId,Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_DcanStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_dcanStatic_verifyConfig (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_McanStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_mcanStatic_verifyConfig (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_MailboxEccTest (Diag_Mailbox_ECC_Cfg         cfg,Diag_Mailbox_ECC_ErrorInfo* ptrErrorInfo,uint32_t* ptrCycleCount);
int32_t MssDiag_TcmEccTest (Diag_TCM_ECC_Cfg cfg, uint32_t* ptrCycleCount);
int32_t MssDiag_TcmParityTest (Diag_TCM_Parity_Cfg cfg, uint32_t* ptrCycleCount);
int32_t MssDiag_EsmStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_EsmStaticCfg_verifyViolationLTCPreload (Diag_StaticCfgErrInfo* ptrErrorInfo,uint32_t* ptrCycleCount);
int32_t MssDiag_RCMStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_rcmStaticCfg_verifyViolation(Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_r4fStaticTest (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_r4fStaticCfg_verifyViolation (Diag_StaticCfgErrInfo* ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_r4fCcmTest (Diag_CCM_Cfg  cfg, Diag_CCM_ErrorInfo *ptrErrorInfo,
                            uint32_t *ptrCycleCount);
int32_t MssDiag_r4fCcmViolationTest(uint8_t instanceId,Diag_CCM_ErrorInfo *ptrErrorInfo,uint32_t* ptrCycleCount);
int32_t MssDiag_dccDiagTest(Diag_DCC_Cfg cfg,Diag_DCC_ErrorInfo *ptrErrorInfo, uint32_t* ptrCycleCount);
int32_t MssDiag_dccViolationTest(uint8_t instanceId,Diag_DCC_ErrorInfo *ptrErrorInfo,
                                 uint32_t* ptrCycleCount);
void MssDiag_BootStatusPrintErrorInfo(const Diag_MSSBootTest_Status* ptrStatus);
int32_t MssDiag_bootTestStatus (uint32_t* ptrCycleCount);
int32_t MssDiag_SelfTest(void);

#endif
