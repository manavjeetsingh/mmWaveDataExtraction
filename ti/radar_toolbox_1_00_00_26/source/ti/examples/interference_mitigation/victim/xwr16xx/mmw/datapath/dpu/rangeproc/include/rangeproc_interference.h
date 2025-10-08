/**
 *   @file  dss_IntfMitg.h
 *
 *   @brief
 *      This is the main header file for the DSS INTF_MITG TI Design.
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
#ifndef DSS_INTF_MITG_H
#define DSS_INTF_MITG_H

 /* MMWAVE Driver Include Files */
#include <ti/common/mmwave_error.h>
#include <ti/drivers/uart/UART.h>
#include <ti/drivers/mailbox/mailbox.h>
#include <ti/drivers/adcbuf/ADCBuf.h>
#include <ti/drivers/edma/edma.h>
#include <ti/drivers/osal/DebugP.h>


/* MMWAVE library Include Files */
#include <ti/control/mmwave/mmwave.h>


#ifdef __cplusplus
extern "C" {
#endif

void intfMitgThresholdComputation(cmplx16ImRe_t inp[restrict], uint32_t len,
	int16_t * pThreshAbs, int16_t * pThreshAbsDiff,
	uint16_t thresholdAbsFac, uint16_t thresholdAbsDiffFac,
	uint16_t thresholdFacBitwidth);

uint32_t intfMitgCreateIIBArr(cmplx16ImRe_t inp[restrict], uint8_t interfDetectArr[restrict], uint32_t len,
	int16_t threshAbs, int16_t threshAbsDiff);

uint32_t intfMitgApplyIIBArr(cmplx16ImRe_t inp[restrict], uint8_t interfDetectArr[restrict], uint32_t len);

inline int16_t jplApprox(cmplx16ImRe_t xl_inp);

uint32_t  intfMitgPerformLinearInterp(cmplx16ImRe_t inp[restrict],  uint32_t len,
	int16_t lastGoodSampleLoc[restrict], int16_t newestGoodSampleLoc[restrict], int32_t numRegionsToRepair);

uint32_t  intfMitgCreateLinearInterpCoeffsFromIIB(uint8_t interfDetectArr[restrict],  uint32_t len,
	int16_t lastGoodSampleLoc[restrict], int16_t newestGoodSampleLoc[restrict]);

#ifdef __cplusplus
}
#endif

#endif /* DSS_INTF_MITG_H */
