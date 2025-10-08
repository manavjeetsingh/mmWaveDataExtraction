/**
 *   @file  dopplerprochwaDDMA.c
 *
 *   @brief
 *      Implements Data path Doppler processing Unit using HWA.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2021 Texas Instruments, Inc.
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

/* MCU+SDK Include files */
#include <kernel/dpl/SemaphoreP.h>
#include <kernel/dpl/HeapP.h>
#include <kernel/dpl/ClockP.h>
#include <kernel/dpl/CacheP.h>
#include <kernel/dpl/CycleCounterP.h>
#include <kernel/dpl/DebugP.h>

/* Utils */
#include <ti/utils/mathutils/mathutils.h>

/* Data Path Include files */
#include <ti/datapath/dpedma/dpedma.h>
#include <ti/datapath/dpedma/dpedmahwa.h>
#include "../dopplerprochwaDDMA.h"
#include "../include/dopplerprochwaDDMAinternal.h"

/* Flag to check input parameters */
#define DEBUG_CHECK_PARAMS   1

/******************************
* DECOMPRESSION STAGE *********
*******************************/

#define DOPPLERPROCHWA_DDMA_DECOMP_NUM_HWA_PARAMSETS        2

#define DECOMP_PING_HWA_PARAMSET_RELATIVE_IDX       0
#define DECOMP_PONG_HWA_PARAMSET_RELATIVE_IDX       1

#define DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PING_IN  0
#define DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PING_OUT 2
#define DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PONG_IN  4
#define DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PONG_OUT 6

#define DPU_DOPPLERHWADDMA_ADDR_DECOMP_PING_IN     HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_DECOMP_PING_OUT    HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_DECOMP_PONG_IN     HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_DECOMP_PONG_OUT    HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PONG_OUT])

/******************************
* DOPPLER STAGE ***************
*******************************/

#define DOPPLERPROCHWA_DDMA_DOPPLER_NUM_HWA_PARAMSETS       12

#define DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PING_IN      0
#define DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PING_OUT     DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PING_IN  
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PING_IN          4
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PING_OUT         DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PING_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PING_IN           0
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PING_OUT          DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_IN      1 
#define DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_OUT     0
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PING_IN           DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PING_OUT
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PING_OUT          0 /* This will actually be stored at an offset in M0 */

#define DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PONG_IN      2   
#define DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PONG_OUT     DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PONG_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PONG_IN          6
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PONG_OUT         DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PONG_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PONG_IN           2 
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PONG_OUT          DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_IN      3
#define DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_OUT     2
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PONG_IN           DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PONG_OUT
#define DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PONG_OUT          2 /* This will actually be stored at an offset in M2 */

#define DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PING_IN      HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PING_OUT     HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_LOGABS_PING_IN          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_LOGABS_PING_OUT         HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_SUMRX_PING_IN           HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_SUMRX_PING_OUT          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PING_IN      HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PING_OUT     HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_SUMTX_PING_IN           HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_SUMTX_PING_OUT          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PING_OUT])

#define DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PONG_IN      HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PONG_OUT     HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PONG_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_LOGABS_PONG_IN          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_LOGABS_PONG_OUT         HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOGABS_PONG_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_SUMRX_PONG_IN           HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_SUMRX_PONG_OUT          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMRX_PONG_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PONG_IN      HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PONG_OUT     HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_SUMTX_PONG_IN           HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_SUMTX_PONG_OUT          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_SUMTX_PONG_OUT])

#define DPU_DOPPLERHWADDMA_DOPPLER_FFT_PING_HWA_PARAMSET_RELATIVE_IDX       1
#define DPU_DOPPLERHWADDMA_LOG_ABS_PING_HWA_PARAMSET_RELATIVE_IDX           2
#define DPU_DOPPLERHWADDMA_SUM_RX_PING_HWA_PARAMSET_RELATIVE_IDX            3
#define DPU_DOPPLERHWADDMA_DDMA_METRIC_PING_HWA_PARAMSET_RELATIVE_IDX       4
#define DPU_DOPPLERHWADDMA_SUM_TX_PING_HWA_PARAMSET_RELATIVE_IDX            5

#define DPU_DOPPLERHWADDMA_DOPPLER_FFT_PONG_HWA_PARAMSET_RELATIVE_IDX       7
#define DPU_DOPPLERHWADDMA_LOG_ABS_PONG_HWA_PARAMSET_RELATIVE_IDX           8
#define DPU_DOPPLERHWADDMA_SUM_RX_PONG_HWA_PARAMSET_RELATIVE_IDX            9
#define DPU_DOPPLERHWADDMA_DDMA_METRIC_PONG_HWA_PARAMSET_RELATIVE_IDX       10
#define DPU_DOPPLERHWADDMA_SUM_TX_PONG_HWA_PARAMSET_RELATIVE_IDX            11

/******************************
* AZIM-CFAR STAGE *************
*******************************/

#define DOPPLERPROCHWA_DDMA_AZIMCFAR_NUM_HWA_PARAMSETS          8

#define DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PING_IN           0
#define DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PING_OUT          DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PING_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PING_IN              4  
#define DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PING_OUT             0
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PING_IN          4
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PING_OUT         1

#define DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PONG_IN           2
#define DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PONG_OUT          DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PONG_IN
#define DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PONG_IN              6  
#define DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PONG_OUT             2               
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PONG_IN          6
#define DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PONG_OUT         3

#define DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PING_IN           HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PING_OUT          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_CFAR_PING_IN              HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PING_IN])  
#define DPU_DOPPLERHWADDMA_ADDR_CFAR_PING_OUT             HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PING_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PING_IN          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PING_IN])
#define DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PING_OUT         HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PING_OUT])

#define DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PONG_IN           HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PONG_OUT          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PONG_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_CFAR_PONG_IN              HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PONG_IN])  
#define DPU_DOPPLERHWADDMA_ADDR_CFAR_PONG_OUT             HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PONG_OUT])
#define DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PONG_IN          HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PONG_IN])
#define DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PONG_OUT         HWADRV_ADDR_TRANSLATE_CPU_TO_HWA(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PONG_OUT])

#define DPU_DOPPLERHWADDMA_AZIMFFT_PING_HWA_PARAMSET_RELATIVE_IDX       1
#define DPU_DOPPLERHWADDMA_CFAR_PING_HWA_PARAMSET_RELATIVE_IDX          2
#define DPU_DOPPLERHWADDMA_LOCALMAX_PING_HWA_PARAMSET_RELATIVE_IDX      3

#define DPU_DOPPLERHWADDMA_AZIMFFT_PONG_HWA_PARAMSET_RELATIVE_IDX       6
#define DPU_DOPPLERHWADDMA_CFAR_PONG_HWA_PARAMSET_RELATIVE_IDX          7
#define DPU_DOPPLERHWADDMA_LOCALMAX_PONG_HWA_PARAMSET_RELATIVE_IDX      8

#define DPU_DOPPLERPROCHWA_PING 0
#define DPU_DOPPLERPROCHWA_PONG 1

#define DPU_DOPPLERPROCHWA_EXTRACT_BIT(n, k) ((n & ( 1 << k )) >> k)

/* User defined heap memory and handle */
#define DOPPLERPROCHWADDMA_HEAP_MEM_SIZE  (sizeof(DPU_DopplerProcHWA_Obj))

static uint8_t gDopplerProcDDMAHeapMem[DOPPLERPROCHWADDMA_HEAP_MEM_SIZE] __attribute__((aligned(HeapP_BYTE_ALIGNMENT)));
static HeapP_Object gDopplerProcDDMAHeapObj;

/*===========================================================
 *                    Internal Functions
 *===========================================================*/

uint32_t gedmaCallback = 0;
/**
 *  @b Description
 *  @n
 *      HWA processing completion call back function.
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 */
static void DPU_DopplerProcHWA_hwaDoneIsrCallback(uint32_t threadIdx, void * arg)
{
    if (arg != NULL) {
        SemaphoreP_post((SemaphoreP_Object*)arg);
    }
}

/**
 *  @b Description
 *  @n
 *      EDMA completion call back function.
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 */
static void DPU_DopplerProcHWA_edmaDoneIsrCallback(Edma_IntrHandle intrHandle,
   void *args)
{
    if (args != NULL) {
        SemaphoreP_post((SemaphoreP_Object*)args);
    }
#ifdef DOPPLER_PROC_DDMA_DPU_DEBUG
    gedmaCallback++;
#endif

}

/**
 *  @b Description
 *  @n
 *      Finds max index in an array
 *
 *  @param[in] arr          - array
 *  @param[in] numSamples   - number of elements in the array
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval max index in the array
 */
int32_t findMaxIdx(int32_t * arr, uint32_t numSamples){

    int32_t i;
    uint32_t maxIdx = 0;

    if(arr == NULL){
        return DPU_DOPPLERPROCHWA_ERROR_FINDMAX;
    }

    for (i = 1; i < numSamples; i++){
        if(arr[i] > arr[maxIdx]){
            maxIdx = i;
        }
    }

    return maxIdx;

}

/**
 *  @b Description
 *  @n
 *      Performs DDMA Demodulation
 *
 *  @param[in] obj          - DPU obj
 *  @param[in] cfg          - DPU configuration
 *  @param[in] blockIdx     - Decompressed block index
 *  @param[in] rangeBinIdx  - Range bin index (starts from 0 for any decompressed block)
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval error code.
 */
int32_t DPU_DopplerProcHWA_DDMADemod(DPU_DopplerProcHWA_Obj      *obj,
                                    DPU_DopplerProcHWA_Config    *cfg,
                                    uint32_t blockIdx,
                                    uint32_t rangeBinIdx)
{
    /*
    DDMA Metric Memory -  [DopplerSubBand][BandIdx]
    For example, for 4 Tx
                    S1  S2  S3  S4  S5  S6 (DDMA Metric)
    DopSubBand1
    DopSubBand2
    ..
    DopSubBandN
    -------------------------------------------------------
    Doppler FFT Memory -  [Doppler Bin][Tx][Rx]
    -------------------------------------------------------
    Pseudocode:
        for i in DopplerSubBand
            find max(DDMAMetric[i][:])
            compute rotIdx
            compute copy indices
            perform copy
    */

    uint32_t numDopplerSubBins;
    int32_t * DDMAMetricMat;
    int32_t * DDMAMetricSubMat;
    uint8_t * dopplerFFTMat;
    uint8_t * destDetSubMat;
    uint8_t * dopMaxSubBandMat;
    int32_t maxIdx;
    uint32_t startIdxTxToCopy1, startIdxTxToCopy2, numTxToCopy1, numTxToCopy2;
    uint32_t bytesPerSample; 
    uint8_t * src;
    uint8_t * dst;
    uint32_t dopSubIdx;
    int32_t retVal = 0;
    
    /* Ping: even rangeBinIdx; Pong: odd rangeBinIdx */
    numDopplerSubBins = obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal;
    bytesPerSample = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    DDMAMetricMat = (int32_t *) cfg->hwRes.DDMAMetricScratchBuf[rangeBinIdx%2]; 
    dopplerFFTMat = (uint8_t *) cfg->hwRes.dopplerFFTScratchBuf[rangeBinIdx%2]; 
    destDetSubMat = (uint8_t *) cfg->hwRes.dopFFTSubMat + 
                                ((blockIdx * obj->decompCfg.rangeBinsPerBlock + rangeBinIdx) * 
                                (cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample
                                 * cfg->staticCfg.numTxAntennas * numDopplerSubBins));
    dopMaxSubBandMat = (uint8_t *)cfg->hwRes.dopMaxSubBandScratchBuf[rangeBinIdx%2]; 

    /* Iterate over DopplerSubBins */
    for (dopSubIdx = 0; dopSubIdx < numDopplerSubBins; dopSubIdx++)
    {        
        DDMAMetricSubMat = &DDMAMetricMat[dopSubIdx * obj->dopplerDemodCfg.numBandsTotal];
        
        /* Find max of DDMA Metric for the particular doppler sub band */
        maxIdx = findMaxIdx(DDMAMetricSubMat, obj->dopplerDemodCfg.numBandsTotal);
        if(maxIdx < 0 || maxIdx >= obj->dopplerDemodCfg.numBandsTotal){
            retVal = DPU_DOPPLERPROCHWA_ERROR_FINDMAX;
            goto exit;
        }
        dopMaxSubBandMat[dopSubIdx] = (uint8_t) maxIdx;

        /* Based on maxIdx, copy could be done in either one or two steps. Two step
        copy happens when there is a wrap-around 
        First copy will be done from startIdxTxToCopy1 index for numTxToCopy1 rows.
        Second copy, if needed, will be done from 0 index for numTxToCopy2 rows.*/
        startIdxTxToCopy1 = maxIdx;
        if ((maxIdx + cfg->staticCfg.numTxAntennas - 1) >= obj->dopplerDemodCfg.numBandsTotal){
            numTxToCopy1 = obj->dopplerDemodCfg.numBandsTotal - maxIdx;
            numTxToCopy2 = cfg->staticCfg.numTxAntennas - numTxToCopy1;
        }
        else{
            numTxToCopy1 = cfg->staticCfg.numTxAntennas;
            numTxToCopy2 = 0;
        }
        startIdxTxToCopy2 = 0;
        
        /* Compute source and dest addresses and perform memory copies */
        
        /* First copy */
        src = ((uint8_t *)dopplerFFTMat);
        src = src + (dopSubIdx * obj->dopplerDemodCfg.numBandsTotal + startIdxTxToCopy1) * cfg->staticCfg.numRxAntennas * bytesPerSample;
        dst = ((uint8_t *)destDetSubMat);
        dst = dst + (dopSubIdx * obj->dopplerDemodCfg.numBandsActive) * cfg->staticCfg.numRxAntennas * bytesPerSample;
        memcpy((void*)dst, (void *)src, cfg->staticCfg.numRxAntennas * bytesPerSample * numTxToCopy1);

        /* Second copy */
        if(numTxToCopy2 > 0){
            src = ((uint8_t *)dopplerFFTMat);
            src = src + (dopSubIdx * obj->dopplerDemodCfg.numBandsTotal + startIdxTxToCopy2) * cfg->staticCfg.numRxAntennas * bytesPerSample;
            dst = ((uint8_t *)destDetSubMat);
            dst = dst + (dopSubIdx * obj->dopplerDemodCfg.numBandsActive) * cfg->staticCfg.numRxAntennas * bytesPerSample + cfg->staticCfg.numRxAntennas * bytesPerSample * numTxToCopy1;
            memcpy((void*)dst, (void *)src, cfg->staticCfg.numRxAntennas * bytesPerSample * numTxToCopy2);
        }

    }

    CacheP_inv(DDMAMetricMat, cfg->hwRes.DDMAMetricScratchBufferSizeBytes/2, CacheP_TYPE_ALLD);

exit:
    return retVal;

}

/**
 *  @b Description
 *  @n
 *      Performs Object List creation
 *
 *  @param[in] obj          - DPU obj
 *  @param[in] cfg          - DPU configuration
 *  @param[in] blockIdx     - Decompressed block index
 *  @param[in] rangeBinIdx  - Range bin index (starts from 0 for any decompressed block)
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval error code.
 */
int32_t DPU_DopplerProcHWA_extractObjectList(DPU_DopplerProcHWA_Obj      *obj,
                                            DPU_DopplerProcHWA_Config    *cfg,
                                            uint32_t blockIdx,
                                            uint32_t rangeBinIdx)
{

    uint8_t * azimFFTMat;
    uint8_t * cfarMat;
    uint8_t * localMaxMat;
    uint8_t * dopSubMaxMat;
    uint8_t * dopFFTMat;
    uint32_t i, numTxAntAzim, numTxAntElev;
    volatile uint32_t numCfarPeaks, DopIdxCurr, AzimIdxCurr, CFARNoiseCurr, numRowsPerAzim;
    volatile uint32_t RowIdx, BitIdx, rowVal, bit, cfarResReal, cfarResImag;
    volatile uint32_t m1Idx, azimPeakSamplem1, azimPeakSample, p1Idx, azimPeakSamplep1, dopFFTMatStartIdx, dopFFTSubMatSizePing;
    DetObjParams * currObjParams;
    int32_t retVal = 0;
    uint16_t cfarPeaksToLoop;
    // uint32_t numObjOut = 0;

    numTxAntAzim = cfg->staticCfg.numAzimTxAntennas;
    numTxAntElev = cfg->staticCfg.numTxAntennas - numTxAntAzim;

    if(rangeBinIdx % 2 == 0){
        numCfarPeaks = obj->numCfarPeaksPing;
    }
    else{
        numCfarPeaks = obj->numCfarPeaksPong;
    }
    // printf("numCfarPeaks = %d\nblockIdx = %d\nrangeBinIdx = %d\n\n", numCfarPeaks, blockIdx, rangeBinIdx);
    // printf("numCfarPeaks = %d\n", numCfarPeaks);

    /* Assign local matrix pointers based on ping/pong */
    azimFFTMat = (uint8_t *) cfg->hwRes.azimFFTScratchBuf[rangeBinIdx%2]; 
    cfarMat = (uint8_t *) cfg->hwRes.cfarScratchBuf[rangeBinIdx%2]; 
    localMaxMat = (uint8_t *) cfg->hwRes.localMaxScratchBuf[rangeBinIdx%2]; 
    dopSubMaxMat = (uint8_t *) cfg->hwRes.dopMaxSubBandScratchBuf[rangeBinIdx%2]; 
    numRowsPerAzim = (obj->cfarAzimFFTCfg.numAzimFFTBins % 32 == 0) ?   
                        (obj->cfarAzimFFTCfg.numAzimFFTBins / 32) : (obj->cfarAzimFFTCfg.numAzimFFTBins / 32) + 1;
    dopFFTSubMatSizePing = cfg->staticCfg.numRxAntennas * cfg->staticCfg.numTxAntennas
                            * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample
                            * obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal;
    dopFFTMat = (uint8_t *) ((uint8_t *) cfg->hwRes.dopFFTSubMat + (rangeBinIdx%2) * dopFFTSubMatSizePing);

    cfarPeaksToLoop = (numCfarPeaks > cfg->hwRes.maxCfarPeaksToDetect)?(cfg->hwRes.maxCfarPeaksToDetect):(numCfarPeaks);
    /* Loop through CFAR peaks and check whether an object is present for a particular peak. If it is present,
       store its parameters in the output list */
    for(i = 0; i < cfarPeaksToLoop; i++){
        
        // printf("i = %d\n", i);
        /* To get the real and imag part */
        cfarResImag = *(uint32_t *)(&cfarMat[i * sizeof(cmplx32ImRe_t)]);
        cfarResReal = *(uint32_t *)(&cfarMat[i * sizeof(cmplx32ImRe_t) + sizeof(cmplx32ImRe_t)/2]);
        CFARNoiseCurr = cfarResImag;
        // printf("CFAR Real = %d, CFAR Imag = %d\n", cfarResReal, cfarResImag);

        AzimIdxCurr = (cfarResReal) >> 12;
        DopIdxCurr = (cfarResReal) - (AzimIdxCurr << 12);

        if(DopIdxCurr > obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal){
            DebugP_assert(0);
        }
        // printf("AzimIdxCurr = %d, DopIdxCurr = %d\n", AzimIdxCurr, DopIdxCurr);

        RowIdx = (DopIdxCurr * numRowsPerAzim) + (AzimIdxCurr / 32);
        BitIdx = AzimIdxCurr - 32 * (AzimIdxCurr / 32);
        // printf("RowIdx = %d, BitIdx = %d\n", RowIdx, BitIdx);

        rowVal = *(uint32_t *)(&localMaxMat[RowIdx * sizeof(uint32_t)]);
        bit = DPU_DOPPLERPROCHWA_EXTRACT_BIT(rowVal, BitIdx);
        // printf("rowVal = %d, bit = %d\n", rowVal, bit);
        // printf("\n");
        
        if(bit){

            // numObjOut++;            

            /* Object found */

            currObjParams = &cfg->hwRes.detObjList[obj->numObjOut];

            currObjParams->azimIdx = AzimIdxCurr;
            currObjParams->dopIdx = DopIdxCurr;
            currObjParams->rangeIdx = blockIdx * obj->decompCfg.rangeBinsPerBlock + rangeBinIdx;
            currObjParams->subBandIdx = (uint32_t )dopSubMaxMat[DopIdxCurr];
            currObjParams->dopIdxActual =  (DopIdxCurr + currObjParams->subBandIdx * obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal)
                                             % (obj->numDopplerBins);
            currObjParams->dopCfarNoise = CFARNoiseCurr;

            dopFFTMatStartIdx = DopIdxCurr * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas * (numTxAntAzim + numTxAntElev);
           
            /* Obtain doppler FFT samples */
            memcpy(&currObjParams->azimSamples, 
                    &dopFFTMat[dopFFTMatStartIdx],
                    numTxAntAzim * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas);
            /* In each doppler bin, Dop FFT corresponding to azim antennas and elev antennas appear one after the other but
                contiguously */
            memcpy(&currObjParams->elevSamples, 
                    &dopFFTMat[dopFFTMatStartIdx + obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas * (numTxAntAzim)],
                    numTxAntElev * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas);

            /* Obtain the azim values at and around the peak */
            m1Idx = (AzimIdxCurr - 1) % obj->cfarAzimFFTCfg.numAzimFFTBins;
            p1Idx = (AzimIdxCurr + 1) % obj->cfarAzimFFTCfg.numAzimFFTBins;
            azimPeakSamplem1 = *(uint16_t *)(&azimFFTMat[(m1Idx + DopIdxCurr * obj->cfarAzimFFTCfg.numAzimFFTBins)
                                             * obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample]);
            azimPeakSample   = *(uint16_t *)(&azimFFTMat[(AzimIdxCurr + DopIdxCurr * obj->cfarAzimFFTCfg.numAzimFFTBins)
                                             * obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample]);
            azimPeakSamplep1 = *(uint16_t *)(&azimFFTMat[(p1Idx + DopIdxCurr * obj->cfarAzimFFTCfg.numAzimFFTBins)
                                             * obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample]);
            currObjParams->azimPeakSamples[0] = azimPeakSamplem1;
            currObjParams->azimPeakSamples[1] = azimPeakSample;
            currObjParams->azimPeakSamples[2] = azimPeakSamplep1;

            obj->numObjOut++;
        }

    }
    // printf("NumObjOut = %d\n\n",numObjOut);
    CacheP_inv(cfarMat, cfg->hwRes.cfarScratchBufferSizeBytes/2, CacheP_TYPE_ALLD);
    

// exit:
    return retVal;
}


/**
 *  @b Description
 *  @n
 *      Calculates DC estimation configuration parameters for HWA
 *
 *  @param[in]  numChirps    - Number of chirps (number of samples that are averaged)
 *  @param[out] dcEstPar     - Pointer to structure with DC estimation parameters
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval None
 */
void DPU_DopplerProcHWA_calcDCEstimParams(uint32_t numChirps, DPU_DopplerProcHWA_DC_estimParams *dcEstPar)
{
    uint32_t shiftCeil;
    uint32_t shiftActual;
    uint32_t prePorcLeftShift;
    uint16_t   dcEstScale;
#define ONEQ8F 256.0

    shiftCeil = mathUtils_ceilLog2(numChirps);

    shiftActual = shiftCeil;
    if (shiftActual < 6)
    {
        shiftActual = 6;
    }
    /* Gain applied in input formatter during the DC estimation time */
    prePorcLeftShift = shiftActual -  shiftCeil;

    dcEstScale = (uint16_t) (ONEQ8F * (float) (1<<shiftCeil) / (float) numChirps + 0.5);

    /* Scale value in Q8 format */
    dcEstPar->dcestScale = dcEstScale;
    /* HWA Programming value */
    dcEstPar->dcestShift = shiftActual - 6;
    /* HWA input formater scale programming value for the DC esimation stage */
    dcEstPar->preProcScaleShift = 8 - prePorcLeftShift;

}

/**
 *  @b Description
 *  @n
 *      Configures HWA for Decompression stage of Doppler processing.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval error code.
 */
static inline int32_t DPU_DopplerProcHWA_configHwaDecompression
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{
    HWA_ParamConfig         hwaParamCfg[DOPPLERPROCHWA_DDMA_DECOMP_NUM_HWA_PARAMSETS];
    HWA_InterruptConfig     paramISRConfig;
    uint32_t                paramsetIdx = 0;
    int32_t                 errCode;
    uint8_t                 destChanPing; //, srcChan;
    uint8_t                 hwParamsetIdx = cfg->hwRes.hwaCfg.decompStageHwaStateMachineCfg.paramSetStartIdx;
    dopplerProcHWADDMADecompressionCfg* pDPDecompParams;

    pDPDecompParams = &obj->decompCfg;

    memset((void*) &hwaParamCfg, 0, DOPPLERPROCHWA_DDMA_DECOMP_NUM_HWA_PARAMSETS * sizeof(HWA_ParamConfig)); 

    /********************************************************************************/

    /*******************************/
    /* PING DECOMPRESSION PARAMSET */
    /*******************************/
{{


    /* adcbuf not mapped, HWA is triggered after edma copy is done */
    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_DMA; 
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx;

    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_COMPRESS;

    /* ACCELMODE CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.ditherEnable = HWA_FEATURE_BIT_ENABLE;  // Enable dither to suppress quantization spurs
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.compressDecompress = HWA_CMP_DCMP_DECOMPRESS;
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.method = HWA_COMPRESS_METHOD_EGE; 
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.passSelect = HWA_COMPRESS_PATHSELECT_BOTHPASSES;
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.headerEnable = HWA_FEATURE_BIT_ENABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.scaleFactorBW = 4; //log2(sample bits) //TODO
    hwaParamCfg[paramsetIdx].accelModeArgs.compressMode.EGEKarrayLength = 3; //log2(8)

    /* SRC CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_DECOMP_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = pDPDecompParams->inputSamplesPerBlock - 1; 
    hwaParamCfg[paramsetIdx].source.srcAIdx = pDPDecompParams->bytesPerSample;  
    hwaParamCfg[paramsetIdx].source.srcBcnt = pDPDecompParams->numBlocksPerPing - 1; 
    hwaParamCfg[paramsetIdx].source.srcBIdx = pDPDecompParams->inputBytesPerBlock;  

    hwaParamCfg[paramsetIdx].source.srcRealComplex = HWA_SAMPLES_FORMAT_COMPLEX;
    hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT;
    hwaParamCfg[paramsetIdx].source.srcSign = HWA_SAMPLES_UNSIGNED;
    hwaParamCfg[paramsetIdx].source.srcConjugate = HWA_FEATURE_BIT_DISABLE;
    hwaParamCfg[paramsetIdx].source.srcScale = 0; 

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_DECOMP_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = pDPDecompParams->outputSamplesPerBlock - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = pDPDecompParams->bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = pDPDecompParams->outputBytesPerBlock; 

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = HWA_SAMPLES_FORMAT_COMPLEX; 
    hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; /* 16 bit real, 16 bit imag */
    hwaParamCfg[paramsetIdx].dest.dstSign = HWA_SAMPLES_SIGNED; 
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 0;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                hwParamsetIdx,
                                &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.decompEdmaCfg.edmaOut.pingPong[0].channel, &destChanPing);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChanPing;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }          
 
}}

    /*******************************/
    /* PONG DECOMPRESSION PARAMSET */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DECOMP_PING_HWA_PARAMSET_RELATIVE_IDX];

    /* adcbuf not mapped, HWA is triggered after edma copy is done */
    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_DMA; 
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; 

    /* SRC CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_DECOMP_PONG_IN; 

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_DECOMP_PONG_OUT; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                hwParamsetIdx,
                                &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }   

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.decompEdmaCfg.edmaOut.pingPong[1].channel, &destChanPing);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChanPing;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }                 
}}

exit:
    return(errCode);
}

/**
 *  @b Description
 *  @n
 *      Configures HWA for Doppler processing and pre-demodulation stage.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval error code.
 */
static inline int32_t DPU_DopplerProcHWA_configHwaDopplerFFTDDMADemod
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{
    HWA_ParamConfig         hwaParamCfg[DOPPLERPROCHWA_DDMA_DOPPLER_NUM_HWA_PARAMSETS];
    uint32_t                paramsetIdx = 0;
    HWA_InterruptConfig     paramISRConfig;
    uint32_t                hwParamsetIdx = cfg->hwRes.hwaCfg.dopplerStageHwaStateMachineCfg.paramSetStartIdx;
    int32_t                 errCode = 0U;
    uint8_t                 destChan;
    uint32_t                fftSizeTemp;
    uint32_t                index;

    memset((void*) &hwaParamCfg, 0, DOPPLERPROCHWA_DDMA_DOPPLER_NUM_HWA_PARAMSETS * sizeof(HWA_ParamConfig));

    /* Disable paramset interrupts */
    for(index = 0; index < DOPPLERPROCHWA_DDMA_DOPPLER_NUM_HWA_PARAMSETS; index++)
    {
        errCode = HWA_disableParamSetInterrupt(obj->hwaHandle, index + hwParamsetIdx, 
                HWA_PARAMDONE_INTERRUPT_TYPE_CPU_INTR1 | HWA_PARAMDONE_INTERRUPT_TYPE_DMA);
        if (errCode != 0)
        {
            goto exit;
        }
    }

    /***********************/
    /* PING DUMMY PARAMSET */
    /***********************/
{

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_DMA;
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; 
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_NONE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }
}

    /*******************************/
    /* PING DOPPLER FFT PARAMSET */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_FFT;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcEstResetMode = HWA_DCEST_INTERFSUM_RESET_MODE_NOUPDATE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcSubEnable = HWA_FEATURE_BIT_DISABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;
    
    /* ACCELMODE CONFIG (FFT) */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftEn = HWA_FEATURE_BIT_ENABLE; 
    if(obj->numDopplerBins % 3 == 0){
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(obj->numDopplerBins/3);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_ENABLE;
    }
    else{
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(obj->numDopplerBins);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_DISABLE;
    }
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.butterflyScaling = 0; //TODO
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.windowEn = HWA_FEATURE_BIT_ENABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.windowStart = cfg->hwRes.hwaCfg.winRamOffset; 
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.winSymm = cfg->hwRes.hwaCfg.winSym;

    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.magLogEn = HWA_FFT_MODE_MAGNITUDE_LOG2_DISABLED;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.fftOutMode = HWA_FFT_MODE_OUTPUT_DEFAULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;

    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = cfg->staticCfg.numChirps - 1; /* this is samples - 1 */
    hwaParamCfg[paramsetIdx].source.srcAIdx = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = cfg->staticCfg.numRxAntennas - 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->dopplerDemodCfg.dopplerIOCfg.input.isReal;
    
    if(obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.dopplerIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT;
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->dopplerDemodCfg.dopplerIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 8;

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = obj->numDopplerBins - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->dopplerDemodCfg.dopplerIOCfg.output.isReal;
    if(obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.dopplerIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{ 
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT;
    }    
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->dopplerDemodCfg.dopplerIOCfg.output.isSigned; 
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 0;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }
  
}}

    /*******************************/
    /* PING LOG ABS PARAMSET */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_FFT;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcEstResetMode = HWA_DCEST_INTERFSUM_RESET_MODE_NOUPDATE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcSubEnable = HWA_FEATURE_BIT_DISABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;
    
    /* ACCELMODE CONFIG (FFT) */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftEn = HWA_FEATURE_BIT_DISABLE; 
    
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.magLogEn = HWA_FFT_MODE_MAGNITUDE_LOG2_ENABLED;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.fftOutMode = HWA_FFT_MODE_OUTPUT_DEFAULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;

    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_LOGABS_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = obj->numDopplerBins * cfg->staticCfg.numRxAntennas - 1; /* this is samples - 1 */
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->dopplerDemodCfg.logAbsIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = 1 - 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = obj->numDopplerBins * cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.logAbsIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->dopplerDemodCfg.logAbsIOCfg.input.isReal;
    if(obj->dopplerDemodCfg.logAbsIOCfg.input.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.logAbsIOCfg.input.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.logAbsIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{ 
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->dopplerDemodCfg.logAbsIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 8; //TODO

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_LOGABS_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = obj->numDopplerBins * cfg->staticCfg.numRxAntennas - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = obj->dopplerDemodCfg.logAbsIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = obj->numDopplerBins * cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.logAbsIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->dopplerDemodCfg.logAbsIOCfg.output.isReal;//HWA_SAMPLES_FORMAT_COMPLEX;
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->dopplerDemodCfg.logAbsIOCfg.output.isSigned; 
    if(obj->dopplerDemodCfg.logAbsIOCfg.output.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.logAbsIOCfg.output.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.logAbsIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{ 
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }    
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 0;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    /* The Doppler FFT EDMA Out is being triggered here instead of after the previous paramset.
       This is because the output of the previous paramset is the input to the current paramset and hence 
       if we performed an EDMA transfer immediately after the previous paramset, the current paramset 
       would not have been able to access the input membank. */
    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDopplerFFTOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    } 

}}

    /*******************************/
    /* PING SUM RX PARAMSET        */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_FFT;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcEstResetMode = HWA_DCEST_INTERFSUM_RESET_MODE_NOUPDATE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcSubEnable = HWA_FEATURE_BIT_DISABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;
    
    /* ACCELMODE CONFIG (FFT) */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftEn = HWA_FEATURE_BIT_ENABLE; 
    
    fftSizeTemp = mathUtils_getValidFFTSize(cfg->staticCfg.numRxAntennas);
    if(fftSizeTemp % 3 == 0){
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp/3);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_ENABLE;
    }
    else{
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_DISABLE;
    }
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.butterflyScaling = 0; //TODO
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.windowEn = HWA_FEATURE_BIT_DISABLE; //HWA_FEATURE_BIT_ENABLE; //TODO

    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.magLogEn = HWA_FFT_MODE_MAGNITUDE_LOG2_DISABLED;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.fftOutMode = HWA_FFT_MODE_OUTPUT_DEFAULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;

    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_SUMRX_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = cfg->staticCfg.numRxAntennas - 1; /* this is samples - 1 */
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->dopplerDemodCfg.sumRxIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = obj->numDopplerBins - 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.sumRxIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->dopplerDemodCfg.sumRxIOCfg.input.isReal;
    if(obj->dopplerDemodCfg.sumRxIOCfg.input.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.sumRxIOCfg.input.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.sumRxIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT;
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->dopplerDemodCfg.sumRxIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 8; //TODO

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_SUMRX_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = 1 - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->dopplerDemodCfg.sumRxIOCfg.output.isReal;
    if(obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.sumRxIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; //TODO
    }
    else{ //TODO Add better check
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT; //TODO
    }
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->dopplerDemodCfg.sumRxIOCfg.output.isSigned;
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = mathUtils_ceilLog2(fftSizeTemp); //TODO
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }
}}

    /*******************************/
    /* PING DDMA METRIC PARAMSET   */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_FFT;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcEstResetMode = HWA_DCEST_INTERFSUM_RESET_MODE_NOUPDATE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcSubEnable = HWA_FEATURE_BIT_DISABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;
    
    /* ACCELMODE CONFIG (FFT) */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftEn = HWA_FEATURE_BIT_ENABLE; /* For sum calculation */
    fftSizeTemp = mathUtils_getValidFFTSize(cfg->staticCfg.numTxAntennas);
    if(fftSizeTemp % 3 == 0){
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp/3);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_ENABLE;
    }
    else{
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_DISABLE;
    }
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.butterflyScaling = 0; //TODO
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.windowEn = HWA_FEATURE_BIT_DISABLE; //HWA_FEATURE_BIT_ENABLE; //TODO

    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.magLogEn = HWA_FFT_MODE_MAGNITUDE_ONLY_ENABLED;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.fftOutMode = HWA_FFT_MODE_OUTPUT_DEFAULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;

    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = obj->dopplerDemodCfg.numBandsActive - 1; /* this is samples - 1 */
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->dopplerDemodCfg.DDMAMetricIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = obj->numDopplerBins - 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = obj->dopplerDemodCfg.DDMAMetricIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->dopplerDemodCfg.DDMAMetricIOCfg.input.isReal;
    if(obj->dopplerDemodCfg.DDMAMetricIOCfg.input.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.DDMAMetricIOCfg.input.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.DDMAMetricIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT; //TODO
    }
    else{ //TODO Add better check
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT; //TODO
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->dopplerDemodCfg.DDMAMetricIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 8; //TODO
    hwaParamCfg[paramsetIdx].source.shuffleMode = HWA_SRC_SHUFFLE_AB_MODE_ADIM;
    hwaParamCfg[paramsetIdx].source.shuffleStart = 0;
    hwaParamCfg[paramsetIdx].source.wrapComb = obj->numDopplerBins * obj->dopplerDemodCfg.DDMAMetricIOCfg.input.bytesPerSample;

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = 1 - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.isReal;
    if(obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.DDMAMetricIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; //TODO
    }
    else{ //TODO Add better check
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT; //TODO
    }
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.isSigned;
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 0;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDDMAMetricOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }   


}}

    /*******************************/
    /* PING SUM TX PARAMSET        */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_SOFTWARE; //HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_FFT;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcEstResetMode = HWA_DCEST_INTERFSUM_RESET_MODE_NOUPDATE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcSubEnable = HWA_FEATURE_BIT_DISABLE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;
    
    /* ACCELMODE CONFIG (FFT) */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftEn = HWA_FEATURE_BIT_ENABLE;
    
    fftSizeTemp = mathUtils_getValidFFTSize(obj->dopplerDemodCfg.numBandsTotal);
    if(fftSizeTemp % 3 == 0){
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp/3);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_ENABLE;
    }
    else{
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_DISABLE;
    }
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.butterflyScaling = 0; //TODO
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.windowEn = HWA_FEATURE_BIT_DISABLE;

    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.magLogEn = HWA_FFT_MODE_MAGNITUDE_ONLY_ENABLED;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.fftOutMode = HWA_FFT_MODE_OUTPUT_DEFAULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_DISABLE;

    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_SUMTX_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = obj->dopplerDemodCfg.numBandsTotal - 1; /* this is samples - 1 */
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->dopplerDemodCfg.sumTxIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) - 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = obj->dopplerDemodCfg.sumTxIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->dopplerDemodCfg.sumTxIOCfg.input.isReal;
    if(obj->dopplerDemodCfg.sumTxIOCfg.input.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.sumTxIOCfg.input.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.sumTxIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT;
    }
    else{ //TODO Add better check
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->dopplerDemodCfg.sumTxIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 8; //TODO
    hwaParamCfg[paramsetIdx].source.shuffleMode = HWA_SRC_SHUFFLE_AB_MODE_ADIM;
    hwaParamCfg[paramsetIdx].source.shuffleStart = 0;
    hwaParamCfg[paramsetIdx].source.wrapComb = obj->numDopplerBins * obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample;

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_SUMTX_PING_OUT + 
                                            obj->numDopplerBins * obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = 1 - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->dopplerDemodCfg.sumTxIOCfg.output.isReal;
    if(obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample == 2 ||
        (obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample == 4 &&
        (!obj->dopplerDemodCfg.sumTxIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; //TODO
    }
    else{ //TODO Add better check
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT; //TODO
    }
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->dopplerDemodCfg.sumTxIOCfg.output.isSigned;
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = mathUtils_ceilLog2(obj->dopplerDemodCfg.numBandsTotal); //TODO IS THIS RIGHT?? or should this be valid ceillog2(fftsize)?
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }   

}}

    /***********************/
    /* PONG DUMMY PARAMSET */
    /***********************/
{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_DMA;
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; //TODO
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_NONE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }
}

    /*******************************/
    /* PONG DOPPLER FFT PARAMSET   */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_DOPPLER_FFT_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_DOPPLERFFT_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }


}}

    /*******************************/
    /* PONG LOG ABS PARAMSET       */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_LOG_ABS_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_LOGABS_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_LOGABS_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    /* The Doppler FFT EDMA Out is being triggered here instead of after the previous paramset.
       This is because the output of the previous paramset is the input to the current paramset and hence 
       if we performed an EDMA transfer immediately after the previous paramset, the current paramset 
       would not have been able to access the input membank. */
    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDopplerFFTOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    } 
    
}}

    /*******************************/
    /* PONG SUM RX PARAMSET        */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_SUM_RX_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_SUMRX_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_SUMRX_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

}}

    /*******************************/
    /* PONG DDMA METRIC PARAMSET   */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_DDMA_METRIC_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_DDMAMETRIC_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDDMAMetricOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }   
    
}}

    /*******************************/
    /* PONG SUM TX PARAMSET        */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_SUM_TX_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_SUMTX_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_SUMTX_PONG_OUT + 
                                            obj->numDopplerBins * obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_SOFTWARE; //HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }   
    
}}

#if 0
    /************ Enable the DMA hookup to this paramset so that data gets copied out ***********/
    /* First, make sure all DMA interrupt/trigger are disabled for this paramset*/
    errCode = HWA_disableParamSetInterrupt(obj->hwaHandle,
                                            cfg->hwRes.hwaCfg.paramSetStartIdx + paramsetIdx,
                                            HWA_PARAMDONE_INTERRUPT_TYPE_DMA | HWA_PARAMDONE_INTERRUPT_TYPE_CPU_INTR1);
    if (errCode != 0)
    {
        goto exit;
    }

    retVal = HWA_getDMAChanIndex(obj->hwaHandle,
                                    cfg->hwRes.edmaCfg.edmaOut.pingPong[pingPongIdx].channel,
                                    &destChan);
    if (retVal != 0)
    {
        goto exit;
    }
    /* Now enable interrupt */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    paramISRConfig.cpu.callbackArg = NULL;
    retVal = HWA_enableParamSetInterrupt(obj->hwaHandle, cfg->hwRes.hwaCfg.paramSetStartIdx + paramsetIdx, &paramISRConfig);
    if (retVal != 0)
    {
        goto exit;
    }
    paramsetIdx++;
#endif


exit:
    return(errCode);
}

/**
 *  @b Description
 *  @n
 *      Configures Azimuth, CFAR, Local Max processing in HWA.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval error code.
 */
static inline int32_t DPU_DopplerProcHWA_configHwaCFARAzimFFT
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{
    HWA_ParamConfig         hwaParamCfg[DOPPLERPROCHWA_DDMA_AZIMCFAR_NUM_HWA_PARAMSETS]; //TODO
    HWA_InterruptConfig     paramISRConfig;
    uint32_t                paramsetIdx = 0;
    uint32_t                hwParamsetIdx = cfg->hwRes.hwaCfg.azimCfarStageHwaStateMachineCfg.paramSetStartIdx;
    int32_t                 errCode = 0U;
    uint8_t                 destChan;
    uint32_t                fftSizeTemp;
    // uint32_t                pingPongIdx;
    // uint32_t                txAntIdx;
    // uint8_t                 triggerMode;
    // uint32_t                addrIdx;
    uint32_t                cfarAvgRight, cfarAvgLeft, cfarGuardCells;

    memset((void*) &hwaParamCfg, 0, DOPPLERPROCHWA_DDMA_AZIMCFAR_NUM_HWA_PARAMSETS * sizeof(HWA_ParamConfig)); //TODO

    /***********************/
    /* PING DUMMY PARAMSET */
    /***********************/
{
    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_DMA;
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; //TODO
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_NONE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }
}

    /*******************************/
    /* PING AZIM FFT PARAMSET      */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_FFT;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcEstResetMode = HWA_DCEST_INTERFSUM_RESET_MODE_NOUPDATE;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.dcSubEnable = HWA_FEATURE_BIT_DISABLE;
    /* Enable complex multiply mode and ensure reading is from common config regs, not the RAM */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.cmultMode = HWA_COMPLEX_MULTIPLY_MODE_VECTOR_MULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.preProcCfg.complexMultiply.modeCfg.vectorMultiplyMode1.cmultScaleEn = HWA_FEATURE_BIT_ENABLE;

    /* ACCELMODE CONFIG (FFT) */
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftEn = HWA_FEATURE_BIT_ENABLE; 
    
    fftSizeTemp = obj->cfarAzimFFTCfg.numAzimFFTBins;
    if(fftSizeTemp % 3 == 0){
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp/3);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_ENABLE;
    }
    else{
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize = mathUtils_ceilLog2(fftSizeTemp);
        hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.fftSize3xEn = HWA_FEATURE_BIT_DISABLE;
    }
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.butterflyScaling = 0; //TODO
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.windowEn = HWA_FEATURE_BIT_DISABLE; /* No windowing at this stage */

    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.magLogEn = HWA_FFT_MODE_MAGNITUDE_LOG2_ENABLED;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.fftOutMode = HWA_FFT_MODE_OUTPUT_DEFAULT;
    hwaParamCfg[paramsetIdx].accelModeArgs.fftMode.postProcCfg.max2Denable = HWA_FEATURE_BIT_ENABLE;

    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = cfg->staticCfg.numRxAntennas * cfg->staticCfg.numAzimTxAntennas - 1; /* this is samples - 1 */
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->cfarAzimFFTCfg.azimFFTIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = obj->numDopplerBins/obj->dopplerDemodCfg.numBandsTotal - 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = cfg->staticCfg.numRxAntennas * cfg->staticCfg.numAzimTxAntennas * obj->cfarAzimFFTCfg.azimFFTIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->cfarAzimFFTCfg.azimFFTIOCfg.input.isReal;
    if(obj->cfarAzimFFTCfg.azimFFTIOCfg.input.bytesPerSample == 2 ||
        (obj->cfarAzimFFTCfg.azimFFTIOCfg.input.bytesPerSample == 4 &&
        (!obj->cfarAzimFFTCfg.azimFFTIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{ //TODO Add better check
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->cfarAzimFFTCfg.azimFFTIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 8; //TODO

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = fftSizeTemp - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = fftSizeTemp * obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isReal;
    if(obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample == 2 ||
        (obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample == 4 &&
        (!obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{ 
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isSigned;
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 0;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

#if 0 /* This cannot be done here since the AzimFFT output data is needed by the subsequent DPUs and hence there's
        a possibility of conflict between HWA and DMA */
    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }  
#endif

}}

    /*******************************/
    /* PING DOPPLER CFAR PARAMSET  */
    /*******************************/
{{

    cfarAvgRight = obj->cfarAzimFFTCfg.cfarCfg.winLen >> 1;
    cfarAvgLeft = obj->cfarAzimFFTCfg.cfarCfg.winLen >> 1;
    cfarGuardCells = obj->cfarAzimFFTCfg.cfarCfg.guardLen;

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_CFAR;

    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.peakGroupEn = obj->cfarAzimFFTCfg.cfarCfg.peakGroupingEn;
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.operMode = HWA_CFAR_OPER_MODE_LOG_INPUT_REAL; /* cfarInpMode = 1, cfarLogMode = 1, cfarAbsMode = 00b */
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.numGuardCells = cfarGuardCells;
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.nAvgDivFactor = obj->cfarAzimFFTCfg.cfarCfg.noiseDivShift;//not applicable in CFAR_OS
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.cyclicModeEn = obj->cfarAzimFFTCfg.cfarCfg.cyclicMode;
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.nAvgMode = obj->cfarAzimFFTCfg.cfarCfg.averageMode;
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.numNoiseSamplesRight = cfarAvgRight;
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.numNoiseSamplesLeft =  cfarAvgLeft;
    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.outputMode = HWA_CFAR_OUTPUT_MODE_I_PEAK_IDX_Q_NEIGHBOR_NOISE_VAL;
    if (obj->cfarAzimFFTCfg.cfarCfg.averageMode == HWA_NOISE_AVG_MODE_CFAR_OS)
	{
	    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.cfarOsKvalue = obj->cfarAzimFFTCfg.cfarCfg.osKvalue;
	    hwaParamCfg[paramsetIdx].accelModeArgs.cfarMode.cfarOsEdgeKScaleEn = obj->cfarAzimFFTCfg.cfarCfg.osEdgeKscaleEn;
	}

    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_CFAR_PING_IN;

    hwaParamCfg[paramsetIdx].source.srcAcnt = obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal - 1
                                + 2 * cfarAvgRight + cfarGuardCells
                                + 2 * cfarAvgLeft + cfarGuardCells;
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->cfarAzimFFTCfg.numAzimFFTBins * obj->cfarAzimFFTCfg.cfarIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBIdx = obj->cfarAzimFFTCfg.cfarIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = obj->cfarAzimFFTCfg.numAzimFFTBins - 1;
    hwaParamCfg[paramsetIdx].source.srcRealComplex = obj->cfarAzimFFTCfg.cfarIOCfg.input.isReal;
    hwaParamCfg[paramsetIdx].source.srcScale = 8;
    if(obj->cfarAzimFFTCfg.cfarIOCfg.input.bytesPerSample == 2 ||
        (obj->cfarAzimFFTCfg.cfarIOCfg.input.bytesPerSample == 4 &&
        (!obj->cfarAzimFFTCfg.cfarIOCfg.input.isReal))){
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_16BIT;
    }
    else{ 
        hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }
    hwaParamCfg[paramsetIdx].source.srcSign = obj->cfarAzimFFTCfg.cfarIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcAcircShift = (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal)
                                                            - (2 * cfarAvgRight + cfarGuardCells);

    if ((obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) % 3 == 0){ /* If numSamples % 3 == 0 */
        hwaParamCfg[paramsetIdx].source.srcCircShiftWrap3 = 1; /* 'b001, means wrap in A dim */
        hwaParamCfg[paramsetIdx].source.srcAcircShiftWrap = mathUtils_ceilLog2((obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) / 3);
    }
    else{
        hwaParamCfg[paramsetIdx].source.srcCircShiftWrap3 = 0;
        hwaParamCfg[paramsetIdx].source.srcAcircShiftWrap = mathUtils_ceilLog2(obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal);
    }

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_CFAR_PING_OUT; 

    hwaParamCfg[paramsetIdx].dest.dstAcnt = cfg->hwRes.maxCfarPeaksToDetect - 1;
    hwaParamCfg[paramsetIdx].dest.dstAIdx = obj->cfarAzimFFTCfg.cfarIOCfg.output.bytesPerSample;
    hwaParamCfg[paramsetIdx].dest.dstBIdx = (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) * obj->cfarAzimFFTCfg.cfarIOCfg.output.bytesPerSample;

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->cfarAzimFFTCfg.cfarIOCfg.output.isReal;
    if(obj->cfarAzimFFTCfg.cfarIOCfg.output.bytesPerSample == 2 ||
        (obj->cfarAzimFFTCfg.cfarIOCfg.output.bytesPerSample == 4 &&
        (!obj->cfarAzimFFTCfg.cfarIOCfg.output.isReal))){
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_16BIT; 
    }
    else{ 
        hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT; 
    }
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->cfarAzimFFTCfg.cfarIOCfg.output.isSigned;
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 8;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaCfarOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }

}}

    /*******************************/
    /* PING LOCAL MAX PARAMSET     */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_LOCALMAX;
    
    /* PREPROC CONFIG */
    hwaParamCfg[paramsetIdx].accelModeArgs.localMaxMode.neighbourBitmask = 85; /* 0 1 0 1 0 1 0 1, "+" shaped comparison */
    hwaParamCfg[paramsetIdx].accelModeArgs.localMaxMode.thresholdBitMask = 0; /* ~ (1 1), enable comparison row wise and column wise */
    hwaParamCfg[paramsetIdx].accelModeArgs.localMaxMode.thresholdMode = 3; /* 1 1, use Max2D internal statistics for thresholding instead of SW based thresholds */
    
    /* SOURCE CONFIG */
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PING_IN; 

    hwaParamCfg[paramsetIdx].source.srcAcnt = 3 - 1; /* Fixed for Local Max */
    hwaParamCfg[paramsetIdx].source.srcAIdx = obj->cfarAzimFFTCfg.numAzimFFTBins * obj->cfarAzimFFTCfg.localMaxIOCfg.input.bytesPerSample;
    hwaParamCfg[paramsetIdx].source.srcBcnt = obj->cfarAzimFFTCfg.numAzimFFTBins / 4 - 1 + 1;
    hwaParamCfg[paramsetIdx].source.srcBIdx = 8; /* Fixed for Local Max */
    hwaParamCfg[paramsetIdx].source.srcCcnt = (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) - 1;
    hwaParamCfg[paramsetIdx].source.srcCIdx = obj->cfarAzimFFTCfg.numAzimFFTBins * obj->cfarAzimFFTCfg.localMaxIOCfg.input.bytesPerSample;
    if ((cfg->staticCfg.numAzimTxAntennas) % 3 == 0){ /* If numSamples % 3 == 0 */
        hwaParamCfg[paramsetIdx].source.srcCircShiftWrap3 = 2; /* 'b020, means wrap in B dim */
        hwaParamCfg[paramsetIdx].source.srcBcircShiftWrap = mathUtils_ceilLog2((obj->cfarAzimFFTCfg.numAzimFFTBins/3) / 4);
    }
    else{
        hwaParamCfg[paramsetIdx].source.srcCircShiftWrap3 = 0;
        hwaParamCfg[paramsetIdx].source.srcBcircShiftWrap = mathUtils_ceilLog2(obj->cfarAzimFFTCfg.numAzimFFTBins / 4);
    }
    hwaParamCfg[paramsetIdx].source.wrapComb = (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal)
                                               * obj->cfarAzimFFTCfg.numAzimFFTBins * obj->cfarAzimFFTCfg.localMaxIOCfg.input.bytesPerSample;

    hwaParamCfg[paramsetIdx].source.srcRealComplex = 0; /* Fixed for Local Max */
    hwaParamCfg[paramsetIdx].source.srcWidth = HWA_SAMPLES_WIDTH_32BIT; /* Fixed for Local Max */
    hwaParamCfg[paramsetIdx].source.srcSign = obj->cfarAzimFFTCfg.localMaxIOCfg.input.isSigned;
    hwaParamCfg[paramsetIdx].source.srcConjugate = 0;
    hwaParamCfg[paramsetIdx].source.srcScale = 0; //TODO

    /* DEST CONFIG */
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PING_OUT; 

    /* dstAcnt = ceil(numAzimFFTBins/32) - 1 
       dstBIdx = 4 * ceil(numAzimFFTBins/32) */
    if(obj->cfarAzimFFTCfg.numAzimFFTBins % 32 == 0){
        hwaParamCfg[paramsetIdx].dest.dstAcnt = obj->cfarAzimFFTCfg.numAzimFFTBins/32 - 1;
        hwaParamCfg[paramsetIdx].dest.dstBIdx = 4 * obj->cfarAzimFFTCfg.numAzimFFTBins/32;
    }
    else{
        hwaParamCfg[paramsetIdx].dest.dstAcnt = obj->cfarAzimFFTCfg.numAzimFFTBins/32 + 1 - 1;
        hwaParamCfg[paramsetIdx].dest.dstBIdx = 4 * (obj->cfarAzimFFTCfg.numAzimFFTBins/32 + 1);
    }
    
    hwaParamCfg[paramsetIdx].dest.dstAIdx = 4; /* Fixed for Local Max */

    hwaParamCfg[paramsetIdx].dest.dstRealComplex = obj->cfarAzimFFTCfg.localMaxIOCfg.output.isReal;
    hwaParamCfg[paramsetIdx].dest.dstWidth = HWA_SAMPLES_WIDTH_32BIT;  /* Fixed for Local Max */
    hwaParamCfg[paramsetIdx].dest.dstSign = obj->cfarAzimFFTCfg.localMaxIOCfg.output.isSigned;
    hwaParamCfg[paramsetIdx].dest.dstConjugate = HWA_FEATURE_BIT_DISABLE; 
    hwaParamCfg[paramsetIdx].dest.dstScale = 0;
    hwaParamCfg[paramsetIdx].dest.dstSkipInit = 0; 

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaLocalMaxOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }

}}

    /* While the AZIM FFT out EDMA trasfer could have been chained to the localmax out EDMA transfer directly, we observed that
       the ISR for the Azim transfer was not being entered. Hence this is a workaround till the issue gets resolved. */
    /*****************************************/
    /* PING DUMMY AZIM FFT TRANSFER PARAMSET */
    /*****************************************/
{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; //TODO
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_NONE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[0].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }  

}

    /***********************/
    /* PONG DUMMY PARAMSET */
    /***********************/
{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_DMA;
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; //TODO
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_NONE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }
}

    /*******************************/
    /* PONG AZIM FFT PARAMSET      */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_AZIMFFT_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_AZIMFFT_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

#if 0
    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }
#endif

}}

    /*******************************/
    /* PONG CFAR-OS PARAMSET       */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_CFAR_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_CFAR_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_CFAR_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaCfarOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }
    
}}

    /*******************************/
    /* PONG LOCAL MAX PARAMSET     */
    /*******************************/
{{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx] = hwaParamCfg[DPU_DOPPLERHWADDMA_LOCALMAX_PING_HWA_PARAMSET_RELATIVE_IDX];
    hwaParamCfg[paramsetIdx].source.srcAddr = DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PONG_IN; 
    hwaParamCfg[paramsetIdx].dest.dstAddr = DPU_DOPPLERHWADDMA_ADDR_LOCALMAX_PONG_OUT;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaLocalMaxOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }

}}

    /* While the AZIM FFT out EDMA trasfer could have been chained to the localmax out EDMA transfer directly, we observed that
       the ISR for the Azim transfer was not being entered. Hence this is a workaround till the issue gets resolved. */
    /*****************************************/
    /* PONG DUMMY AZIM FFT TRANSFER PARAMSET */
    /*****************************************/
{

    paramsetIdx++;
    hwParamsetIdx++;

    hwaParamCfg[paramsetIdx].triggerMode = HWA_TRIG_MODE_IMMEDIATE;
    hwaParamCfg[paramsetIdx].triggerSrc = hwParamsetIdx; //TODO
    hwaParamCfg[paramsetIdx].accelMode = HWA_ACCELMODE_NONE;

    errCode = HWA_configParamSet(obj->hwaHandle,
                                  hwParamsetIdx,
                                  &hwaParamCfg[paramsetIdx],NULL);
    if (errCode != 0)
    {
        goto exit;
    }

    errCode = HWA_getDMAChanIndex(obj->hwaHandle, cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[1].channel, &destChan);
    if (errCode != 0)
    {
        goto exit;
    }
    /* enable the DMA hookup to this paramset so that data gets copied out */
    paramISRConfig.interruptTypeFlag = HWA_PARAMDONE_INTERRUPT_TYPE_DMA;
    paramISRConfig.dma.dstChannel = destChan;
    errCode = HWA_enableParamSetInterrupt(obj->hwaHandle, hwParamsetIdx, &paramISRConfig);
    if (errCode != 0)
    {
        goto exit;
    }  

}

exit:
    return(errCode);
}

/**
 *  @b Description
 *  @n
 *  Doppler DPU EDMA configuration that sends Doppler FFT In data (decompression out data)
 *  from L3 to HWA memory
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
int32_t DPU_DopplerProcHWA_configEdmaDopplerIn
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
){

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    /* Program EDMA Data in from Decompressed Radar Cube scratch buffer to HWA Memory */
    /* PING */
    {{
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaInSignature.pingPong[0].channel;
    chainingCfg.isIntermediateChainingEnabled = true; //TODO
    chainingCfg.isFinalChainingEnabled        = true;

    syncABCfg.srcAddress  = (uint32_t)(cfg->hwRes.decompScratchBuf); //TODO
    syncABCfg.destAddress = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PING_IN]);
    syncABCfg.aCount      = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample;
    syncABCfg.bCount      = cfg->staticCfg.numChirps;
    syncABCfg.cCount      = obj->decompCfg.rangeBinsPerBlock / 2; /* Ping and Pong */
    syncABCfg.srcBIdx     = cfg->staticCfg.numRxAntennas * obj->decompCfg.rangeBinsPerBlock * obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample;
    syncABCfg.dstBIdx     = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample;
    syncABCfg.srcCIdx     = obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample * cfg->staticCfg.numRxAntennas * 2; /* Ping and Pong */
    syncABCfg.dstCIdx     = 0; /* One range bin in a block is processed at a time */

    retVal = DPEDMA_configSyncAB(cfg->hwRes.edmaCfg.edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIn.pingPong[0], //TODO
                                    &chainingCfg,
                                    &syncABCfg,
                                    false,//isEventTriggered
                                    false, //isIntermediateTransferCompletionEnabled
                                    false,//isTransferCompletionEnabled
                                    NULL, //transferCompletionCallbackFxn
                                    NULL,
                                    NULL);//transferCompletionCallbackFxnArg

    /* One Hot Signature to trigger the HWA */
    retVal = DPEDMAHWA_configOneHotSignature(cfg->hwRes.edmaCfg.edmaHandle, 
                                                  &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaInSignature.pingPong[0],
                                                  obj->hwaHandle,
                                                  obj->dopplerDemodCfg.hwaDmaTriggerSourcePingPongIn[0],
                                                  false);    
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }
    }}

    /* PONG */
    {{
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaInSignature.pingPong[1].channel;
    syncABCfg.srcAddress  = (uint32_t)((uint8_t *)cfg->hwRes.decompScratchBuf + 
                                        obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample * cfg->staticCfg.numRxAntennas); //TODO
    syncABCfg.destAddress = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PONG_IN]);
    retVal = DPEDMA_configSyncAB(cfg->hwRes.edmaCfg.edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIn.pingPong[1],
                                    &chainingCfg,
                                    &syncABCfg,
                                    false,//isEventTriggered
                                    false, //TODO true, //isIntermediateTransferCompletionEnabled
                                    false,//isTransferCompletionEnabled
                                    NULL, //transferCompletionCallbackFxn
                                    NULL,//transferCompletionCallbackFxnArg
                                    NULL);

    /* One Hot Signature to trigger the HWA */
    retVal = DPEDMAHWA_configOneHotSignature(cfg->hwRes.edmaCfg.edmaHandle, 
                                                  &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaInSignature.pingPong[1],
                                                  obj->hwaHandle,
                                                  obj->dopplerDemodCfg.hwaDmaTriggerSourcePingPongIn[1],
                                                  false);
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }
    
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  Doppler DPU EDMA configuration that sends compressed radar cube data to HWA memory for 
 *  decompression.
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
int32_t DPU_DopplerProcHWA_configEdmaDecompressionIn
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
){

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.decompEdmaCfg.edmaInSignature.pingPong[0].channel;
    chainingCfg.isIntermediateChainingEnabled = true; //TODO
    chainingCfg.isFinalChainingEnabled        = true;

    syncABCfg.srcAddress  = (uint32_t)(obj->decompCfg.decompEdmaToHwaStartAddress); //TODO
    syncABCfg.destAddress = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PING_IN]);
    syncABCfg.aCount      = obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing;
    syncABCfg.bCount      = 1;
    syncABCfg.cCount      = obj->decompCfg.numLoops;
    syncABCfg.srcBIdx     = obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.dstBIdx     = obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.srcCIdx     = obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.dstCIdx     = 0;

    retVal = DPEDMA_configSyncAB(cfg->hwRes.edmaCfg.edmaHandle,
                                    &cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[0], //TODO
                                    &chainingCfg,
                                    &syncABCfg,
                                    false,//isEventTriggered
                                    false, //isIntermediateTransferCompletionEnabled
                                    false,//isTransferCompletionEnabled
                                    NULL, //transferCompletionCallbackFxn
                                    NULL, //transferCompletionCallbackFxnArg
                                    NULL); /* intrObj */

    /* One Hot Signature to trigger the HWA */
    retVal = DPEDMAHWA_configOneHotSignature(cfg->hwRes.edmaCfg.edmaHandle, 
                                                  &cfg->hwRes.edmaCfg.decompEdmaCfg.edmaInSignature.pingPong[0],
                                                  obj->hwaHandle,
                                                  obj->decompCfg.hwaDmaTriggerSourcePingPongIn[0],
                                                  false);
    
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.decompEdmaCfg.edmaInSignature.pingPong[1].channel;
    syncABCfg.srcAddress  = (uint32_t)(((uint8_t *)obj->decompCfg.decompEdmaToHwaStartAddress) + 
                                        obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing); //TODO
    syncABCfg.destAddress = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PONG_IN]);
    retVal = DPEDMA_configSyncAB(cfg->hwRes.edmaCfg.edmaHandle,
                                    &cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[1],
                                    &chainingCfg,
                                    &syncABCfg,
                                    false,//isEventTriggered
                                    false, //TODO true, //isIntermediateTransferCompletionEnabled
                                    false,//isTransferCompletionEnabled
                                    NULL, //transferCompletionCallbackFxn
                                    NULL, //transferCompletionCallbackFxnArg
                                    NULL); /* intrObj */

    /* One Hot Signature to trigger the HWA */
    retVal = DPEDMAHWA_configOneHotSignature(cfg->hwRes.edmaCfg.edmaHandle, 
                                                  &cfg->hwRes.edmaCfg.decompEdmaCfg.edmaInSignature.pingPong[1],
                                                  obj->hwaHandle,
                                                  obj->decompCfg.hwaDmaTriggerSourcePingPongIn[1],
                                                  false);
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }
    
    //}}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  Doppler DPU EDMA configuration for Decompressed data out of HWA into L3.
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaDecompressionOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[0].channel; //todo
    chainingCfg.isIntermediateChainingEnabled = true; //TODO false;
    chainingCfg.isFinalChainingEnabled        = false; //true; //TODO

    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PING_OUT]);
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.decompScratchBuf);
    syncABCfg.aCount      = obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing;
    syncABCfg.bCount      = 1;
    syncABCfg.cCount      = obj->decompCfg.numLoops;
    syncABCfg.srcBIdx     = 0; //obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.dstBIdx     = obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.srcCIdx     = 0; //obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.dstCIdx     = obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2; //0; //TODO

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.decompEdmaCfg.edmaOut.pingPong[0], //TODO
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered //TODO
                                    false, //isIntermediateTransferCompletionEnabled
                                    false,//isTransferCompletionEnabled //TODO
                                    NULL, //transferCompletionCallbackFxn
                                    NULL, //transferCompletionCallbackFxnArg
                                    NULL); /* intrObj */

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[1].channel;
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DECOMP_PONG_OUT]); //TODO
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.decompScratchBuf + 
                                    obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing);
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.decompEdmaCfg.edmaOut.pingPong[1], //TODO
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered //TODO
                                    false, //isIntermediateTransferCompletionEnabled
                                    true,//isTransferCompletionEnabled
                                    DPU_DopplerProcHWA_edmaDoneIsrCallback, //transferCompletionCallbackFxn
                                    (void *)((uint32_t)&obj->decompEdmaOutDoneSemaHandle), //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIntrObjDecompOut); /* intrObj */
    
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  Doppler DPU EDMA configuration for sending out Doppler FFT data from HWA
 *  Memory to L2.
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaDopplerFFTOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {DPU_DopplerProcHWA_edmaDoneIsrCallback, DPU_DopplerProcHWA_edmaDoneIsrCallback};
    uint32_t            doneCllbackFuncArg[2] = {(uint32_t)&obj->dopplerFFTPingEdmaOutDoneSemaHandle, (uint32_t)&obj->dopplerFFTPongEdmaOutDoneSemaHandle};
    bool                doneTransferCompletionEnabled[2] = {true, true};

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDopplerFFTOut.pingPong[DPU_DOPPLERPROCHWA_PING].channel; //todo
    chainingCfg.isIntermediateChainingEnabled = true; 
    chainingCfg.isFinalChainingEnabled        = false; 

    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PING_OUT]);
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.dopplerFFTScratchBuf[DPU_DOPPLERPROCHWA_PING]);
    syncABCfg.aCount      = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas; // * obj->numDopplerBins;
    syncABCfg.bCount      = obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal; //1;
    syncABCfg.cCount      = obj->dopplerDemodCfg.numBandsTotal; //obj->decompCfg.rangeBinsPerBlock / 2;
    syncABCfg.srcBIdx     = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;//obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas * 2;
    syncABCfg.dstBIdx     = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.numBandsTotal;//2;
    syncABCfg.srcCIdx     = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas
                            * obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal;
    syncABCfg.dstCIdx     = cfg->staticCfg.numRxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDopplerFFTOut.pingPong[DPU_DOPPLERPROCHWA_PING], //TODO
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered // UPON HAWA COMPLETION
                                    false, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled //TODO
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIntrObjDopplerFFTOut.pingPong[DPU_DOPPLERPROCHWA_PING]); /* intrObj */

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDopplerFFTOut.pingPong[1].channel; //TODO
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DOPPLERFFT_PONG_OUT]); //TODO
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.dopplerFFTScratchBuf[1]); //((uint8_t *)cfg->hwRes.detMatrix.data + //TODO should be uint8_t*??
                                    // obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample* cfg->staticCfg.numRxAntennas * obj->numDopplerBins);
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDopplerFFTOut.pingPong[1], //TODO
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered //TODO
                                    false, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[1],//isTransferCompletionEnabled //TODO
                                    doneCllbackFunc[1], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[1], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIntrObjDopplerFFTOut.pingPong[DPU_DOPPLERPROCHWA_PONG]); /* intrObj */
    
    }}

exit:
    return(retVal);

}


/**
 *  @b Description
 *  @n
 *  EDMA Configuration to send DDMA Metric data out to L2 from HWA
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaDDMAMetricOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {DPU_DopplerProcHWA_edmaDoneIsrCallback, DPU_DopplerProcHWA_edmaDoneIsrCallback};
    uint32_t            doneCllbackFuncArg[2] = {(uint32_t)&obj->DDMAMetricPingEdmaOutDoneSemaHandle, (uint32_t)&obj->DDMAMetricPongEdmaOutDoneSemaHandle};
    bool                doneTransferCompletionEnabled[2] = {true, true};

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDDMAMetricOut.pingPong[DPU_DOPPLERPROCHWA_PING].channel; 
    chainingCfg.isIntermediateChainingEnabled = true;
    chainingCfg.isFinalChainingEnabled        = false;
    
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_OUT]);
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.DDMAMetricScratchBuf[DPU_DOPPLERPROCHWA_PING]); 
    syncABCfg.aCount      = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample;
    syncABCfg.bCount      = obj->dopplerDemodCfg.numBandsTotal;
    syncABCfg.cCount      = obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal;
    syncABCfg.srcBIdx     = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal); //obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample * cfg->staticCfg.numRxAntennas * 2;
    syncABCfg.dstBIdx     = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample; // * cfg->staticCfg.numRxAntennas * 2;
    syncABCfg.srcCIdx     = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample; //obj->decompCfg.outputBytesPerBlock * obj->decompCfg.numChirpsPerPing * 2;
    syncABCfg.dstCIdx     = obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample * obj->dopplerDemodCfg.numBandsTotal;

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDDMAMetricOut.pingPong[DPU_DOPPLERPROCHWA_PING], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered // UPON HAWA COMPLETION
                                    false, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIntrObjDDMAMetricOut.pingPong[DPU_DOPPLERPROCHWA_PING]); /* Interrupt object */

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDDMAMetricOut.pingPong[DPU_DOPPLERPROCHWA_PONG].channel; 
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_OUT]); 
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.DDMAMetricScratchBuf[DPU_DOPPLERPROCHWA_PONG]);
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaDDMAMetricOut.pingPong[DPU_DOPPLERPROCHWA_PONG], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered 
                                    false, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PONG],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIntrObjDDMAMetricOut.pingPong[DPU_DOPPLERPROCHWA_PONG]); /* Interrupt object */
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  Doppler DPU EDMA configuration to transfer sumTx data from HWA Memory to L3.
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaDopplerFFTSumTxOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncACfg            syncACfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {DPU_DopplerProcHWA_edmaDoneIsrCallback, DPU_DopplerProcHWA_edmaDoneIsrCallback};
    uint32_t            doneCllbackFuncArg[2] = {(uint32_t)&obj->sumLogAbsPingEdmaOutDoneSemaHandle, (uint32_t)&obj->sumLogAbsPongEdmaOutDoneSemaHandle};
    bool                doneTransferCompletionEnabled[2] = {true, true};

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }


    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[DPU_DOPPLERPROCHWA_PING].channel; //todo
    chainingCfg.isIntermediateChainingEnabled = false;
    chainingCfg.isFinalChainingEnabled        = false;
    
    syncACfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PING_OUT]
                                    + (obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample * obj->numDopplerBins));
    syncACfg.destAddress = (uint32_t)(cfg->hwRes.detMatrix.data);
    syncACfg.aCount      = obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample
                            * obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal;
    syncACfg.bCount      = obj->decompCfg.rangeBinsPerBlock / 2;
    syncACfg.cCount      = obj->decompCfg.numBlocks;
    syncACfg.srcBIdx     = 0;
    syncACfg.dstBIdx     = obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample
                            * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal)
                            * 2;
    syncACfg.srcCIdx     = 0;
    syncACfg.dstCIdx     = obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample
                            * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal)
                            * 2;

    retVal = DPEDMA_configSyncA(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[DPU_DOPPLERPROCHWA_PING], //TODO
                                    &chainingCfg,
                                    &syncACfg,
                                    true,//isEventTriggered // UPON HAWA COMPLETION
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled //TODO
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIntrObjSumtxOut.pingPong[DPU_DOPPLERPROCHWA_PING]);

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[DPU_DOPPLERPROCHWA_PONG].channel; //TODO
    syncACfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_DDMAMETRIC_PONG_OUT]
                                     + (obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample * obj->numDopplerBins));
    syncACfg.destAddress = (uint32_t)((uint8_t *)cfg->hwRes.detMatrix.data
                                     + (obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample
                                         * obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal)); //TODO
    retVal = DPEDMA_configSyncA(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaSumLogAbsOut.pingPong[DPU_DOPPLERPROCHWA_PONG], //TODO
                                    &chainingCfg,
                                    &syncACfg,
                                    true,//isEventTriggered //TODO
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PONG],//isTransferCompletionEnabled //TODO
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIntrObjSumtxOut.pingPong[DPU_DOPPLERPROCHWA_PONG]); /* Interrupt object */
    
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  Transfers Doppler FFT Output (Azimuth FFT Input) to HWA memory for Azimuth FFT Processing
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaAzimFFTIn
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg,
    uint32_t                    srcAddr
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {NULL, NULL};
    uint32_t            doneCllbackFuncArg[2] = {NULL, NULL};
    bool                doneTransferCompletionEnabled[2] = {false, false};

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTInSignature.pingPong[DPU_DOPPLERPROCHWA_PING].channel;
    chainingCfg.isIntermediateChainingEnabled = true;
    chainingCfg.isFinalChainingEnabled        = true;
    
    syncABCfg.srcAddress  = srcAddr;
    syncABCfg.destAddress = obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PING_IN];
    syncABCfg.aCount      = cfg->staticCfg.numRxAntennas * cfg->staticCfg.numAzimTxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    syncABCfg.bCount      = obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal;
    syncABCfg.cCount      = obj->decompCfg.rangeBinsPerBlock / 2;
    syncABCfg.srcBIdx     = cfg->staticCfg.numRxAntennas * cfg->staticCfg.numTxAntennas 
                            * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    syncABCfg.dstBIdx     = cfg->staticCfg.numRxAntennas * cfg->staticCfg.numAzimTxAntennas * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    syncABCfg.srcCIdx     = 0; // cfg->staticCfg.numRxAntennas * cfg->staticCfg.numTxAntennas 
    //                         * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample 
    //                         * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) 
    //                         * 2;
    syncABCfg.dstCIdx     = 0;

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTIn.pingPong[DPU_DOPPLERPROCHWA_PING],
                                    &chainingCfg,
                                    &syncABCfg,
                                    false,//isEventTriggered // UPON HAWA COMPLETION
                                    false, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                    NULL); /* Interrupt object */
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* One Hot Signature to trigger the HWA */
    retVal = DPEDMAHWA_configOneHotSignature(cfg->hwRes.edmaCfg.edmaHandle, 
                                                  &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTInSignature.pingPong[DPU_DOPPLERPROCHWA_PING],
                                                  obj->hwaHandle,
                                                  obj->cfarAzimFFTCfg.hwaDmaTriggerSourcePingPongIn[DPU_DOPPLERPROCHWA_PING],
                                                  false);
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }


    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTInSignature.pingPong[DPU_DOPPLERPROCHWA_PONG].channel;
    syncABCfg.srcAddress  = (uint32_t)((uint8_t *)srcAddr + 
                            cfg->staticCfg.numRxAntennas * cfg->staticCfg.numTxAntennas
                             * obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample
                             * obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal);
    syncABCfg.destAddress = obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PONG_IN];
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTIn.pingPong[DPU_DOPPLERPROCHWA_PONG],
                                    &chainingCfg,
                                    &syncABCfg,
                                    false,//isEventTriggered 
                                    false, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PONG],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxnArg
                                    NULL); /* Interrupt object */

    /* One Hot Signature to trigger the HWA */
    retVal = DPEDMAHWA_configOneHotSignature(cfg->hwRes.edmaCfg.edmaHandle, 
                                                &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTInSignature.pingPong[DPU_DOPPLERPROCHWA_PONG],
                                                obj->hwaHandle,
                                                obj->cfarAzimFFTCfg.hwaDmaTriggerSourcePingPongIn[DPU_DOPPLERPROCHWA_PONG],
                                                false);
    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }
    
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  EDMA Configuration to send Azimuth FFT data out to L2 from HWA
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaAzimFFTOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {DPU_DopplerProcHWA_edmaDoneIsrCallback, DPU_DopplerProcHWA_edmaDoneIsrCallback};
    uint32_t            doneCllbackFuncArg[2] = {(uint32_t)&obj->azimFFTPingEdmaOutDoneSemaHandle, (uint32_t)&obj->azimFFTPongEdmaOutDoneSemaHandle};
    bool                doneTransferCompletionEnabled[2] = {true, true};
    uint32_t    azimFFTOutSize;

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    azimFFTOutSize = obj->cfarAzimFFTCfg.numAzimFFTBins
                    * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) 
                    * obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample;


    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[DPU_DOPPLERPROCHWA_PING].channel; 
    chainingCfg.isIntermediateChainingEnabled = false;
    chainingCfg.isFinalChainingEnabled        = false;
    
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PING_OUT]);
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.azimFFTScratchBuf[DPU_DOPPLERPROCHWA_PING]); 
    syncABCfg.aCount      = azimFFTOutSize;
    syncABCfg.bCount      = 1;
    syncABCfg.cCount      = 1;
    syncABCfg.srcBIdx     = azimFFTOutSize * 2;
    syncABCfg.dstBIdx     = azimFFTOutSize * 2;
    syncABCfg.srcCIdx     = 0;
    syncABCfg.dstCIdx     = 0;

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[DPU_DOPPLERPROCHWA_PING], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered 
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaIntrObjAzimFFTOut.pingPong[DPU_DOPPLERPROCHWA_PING]); /* intrObj */

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[DPU_DOPPLERPROCHWA_PONG].channel; 
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_AZIMFFT_PONG_OUT]); 
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.azimFFTScratchBuf[DPU_DOPPLERPROCHWA_PONG]);
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTOut.pingPong[DPU_DOPPLERPROCHWA_PONG], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered 
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PONG],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaIntrObjAzimFFTOut.pingPong[DPU_DOPPLERPROCHWA_PONG]); /* intrObj */
    
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  EDMA Configuration to send CFAR data out to L2 from HWA
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaCfarOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {DPU_DopplerProcHWA_edmaDoneIsrCallback, DPU_DopplerProcHWA_edmaDoneIsrCallback};
    uint32_t            doneCllbackFuncArg[2] = {(uint32_t)&obj->cfarPingEdmaOutDoneSemaHandle, (uint32_t)&obj->cfarPongEdmaOutDoneSemaHandle};
    bool                doneTransferCompletionEnabled[2] = {true, true};

    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaCfarOut.pingPong[DPU_DOPPLERPROCHWA_PING].channel; 
    chainingCfg.isIntermediateChainingEnabled = false;
    chainingCfg.isFinalChainingEnabled        = false;
    
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PING_OUT]);
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.cfarScratchBuf[DPU_DOPPLERPROCHWA_PING]);
    syncABCfg.aCount      = cfg->hwRes.maxCfarPeaksToDetect * obj->cfarAzimFFTCfg.cfarIOCfg.output.bytesPerSample; 
    syncABCfg.bCount      = 1;
    syncABCfg.cCount      = 1;
    syncABCfg.srcBIdx     = DECOMP_HWA_MEMBANK_SIZE * 2 - 1; 
    syncABCfg.dstBIdx     = DECOMP_HWA_MEMBANK_SIZE * 2 - 1; 
    syncABCfg.srcCIdx     = 0; 
    syncABCfg.dstCIdx     = 0;

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaCfarOut.pingPong[DPU_DOPPLERPROCHWA_PING], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered // UPON HAWA COMPLETION
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                     cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaIntrObjCfarOut.pingPong[DPU_DOPPLERPROCHWA_PING]); /* intrObj */

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaCfarOut.pingPong[DPU_DOPPLERPROCHWA_PONG].channel; 
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_CFAR_PONG_OUT]); 
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.cfarScratchBuf[DPU_DOPPLERPROCHWA_PONG]);
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaCfarOut.pingPong[DPU_DOPPLERPROCHWA_PONG], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered 
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PONG],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxnArg
                                     cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaIntrObjCfarOut.pingPong[DPU_DOPPLERPROCHWA_PONG]); /* intrObj */
    
    }}

exit:
    return(retVal);

}

/**
 *  @b Description
 *  @n
 *  EDMA Configuration to send Local Max data out to L2 from HWA
 *  This implementation of doppler processing involves Ping/Pong 
 *  Mechanism, hence there are two sets of EDMA transfer.
 *
 *  @param[in] obj    - DPU obj
 *  @param[in] cfg    - DPU configuration
 *
 *  \ingroup    DPU_DOPPLERPROC_INTERNAL_FUNCTION
 *
 *  @retval EDMA error code, see EDMA API.
 */
static inline int32_t DPU_DopplerProcHWA_configEdmaLocalMaxOut
(
    DPU_DopplerProcHWA_Obj      *obj,
    DPU_DopplerProcHWA_Config   *cfg
)
{

    DPEDMA_syncABCfg            syncABCfg;
    DPEDMA_ChainingCfg          chainingCfg;
    int32_t                     retVal;
    Edma_EventCallback doneCllbackFunc[2] = {DPU_DopplerProcHWA_edmaDoneIsrCallback, DPU_DopplerProcHWA_edmaDoneIsrCallback};
    uint32_t            doneCllbackFuncArg[2] = {(uint32_t)&obj->localMaxPingEdmaOutDoneSemaHandle, (uint32_t)&obj->localMaxPongEdmaOutDoneSemaHandle};
    bool                doneTransferCompletionEnabled[2] = {true, true};
    uint32_t    localMaxOutSize;
    
    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }
    
    if(obj->cfarAzimFFTCfg.numAzimFFTBins % 32 == 0){
        localMaxOutSize = (obj->cfarAzimFFTCfg.numAzimFFTBins / 32) 
                            * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) * 4;
    }
    else{
        localMaxOutSize = ((obj->cfarAzimFFTCfg.numAzimFFTBins / 32) + 1) 
                            * (obj->numDopplerBins / obj->dopplerDemodCfg.numBandsTotal) * 4;
    }    

    {{
    
    /* PING */
    chainingCfg.chainingChannel                  = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaLocalMaxOut.pingPong[DPU_DOPPLERPROCHWA_PING].channel; //todo
    chainingCfg.isIntermediateChainingEnabled = false;
    chainingCfg.isFinalChainingEnabled        = false;
    
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PING_OUT]);
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.localMaxScratchBuf[DPU_DOPPLERPROCHWA_PING]);
    syncABCfg.aCount      = localMaxOutSize;
    syncABCfg.bCount      = 1;
    syncABCfg.cCount      = 1;
    syncABCfg.srcBIdx     = localMaxOutSize * 2;
    syncABCfg.dstBIdx     = localMaxOutSize * 2;
    syncABCfg.srcCIdx     = 0;
    syncABCfg.dstCIdx     = 0;

    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaLocalMaxOut.pingPong[DPU_DOPPLERPROCHWA_PING],
                                    &chainingCfg,
                                    &syncABCfg,
                                    true, // UPON HAWA COMPLETION
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PING],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PING], //transferCompletionCallbackFxnArg
                                     cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaIntrObjLocalMaxOut.pingPong[DPU_DOPPLERPROCHWA_PING]); /* intrObj */

    if (retVal != SystemP_SUCCESS)
    {
        goto exit;
    }

    /* PONG */
    chainingCfg.chainingChannel = cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaLocalMaxOut.pingPong[DPU_DOPPLERPROCHWA_PONG].channel; 
    syncABCfg.srcAddress  = (uint32_t)(obj->hwaMemBankAddr[DPU_DOPPLERHWADDMA_MEM_BANK_LOCALMAX_PONG_OUT]); 
    syncABCfg.destAddress = (uint32_t)(cfg->hwRes.localMaxScratchBuf[DPU_DOPPLERPROCHWA_PONG]);
    retVal = DPEDMA_configSyncAB(   obj->edmaHandle,
                                    &cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaLocalMaxOut.pingPong[DPU_DOPPLERPROCHWA_PONG], 
                                    &chainingCfg,
                                    &syncABCfg,
                                    true,//isEventTriggered 
                                    true, //isIntermediateTransferCompletionEnabled
                                    doneTransferCompletionEnabled[DPU_DOPPLERPROCHWA_PONG],//isTransferCompletionEnabled 
                                    doneCllbackFunc[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxn
                                    (void *)doneCllbackFuncArg[DPU_DOPPLERPROCHWA_PONG], //transferCompletionCallbackFxnArg
                                    cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaIntrObjLocalMaxOut.pingPong[DPU_DOPPLERPROCHWA_PONG]); /* intrObj */
    
    }}

exit:
    return(retVal);

}



/*===========================================================
 *                    Doppler Proc External APIs
 *===========================================================*/

/**
 *  @b Description
 *  @n
 *      dopplerProc DPU init function. It allocates memory to store
 *  its internal data object and returns a handle if it executes successfully.
 *
 *  @param[in]   initCfg Pointer to initial configuration parameters
 *  @param[out]  errCode Pointer to errCode generates by the API
 *
 *  \ingroup    DPU_DOPPLERPROC_EXTERNAL_FUNCTION
 *
 *  @retval
 *      Success     - valid handle
 *  @retval
 *      Error       - NULL
 */
DPU_DopplerProcHWA_Obj * gDebugDopplerProcHwaObj = (DPU_DopplerProcHWA_Obj *)&gDopplerProcDDMAHeapMem[0];
DPU_DopplerProcHWA_Handle DPU_DopplerProcHWA_init
(
    DPU_DopplerProcHWA_InitParams *initCfg,
    int32_t                    *errCode
)
{
    DPU_DopplerProcHWA_Obj  *obj = NULL;
    HWA_MemInfo             hwaMemInfo;
    uint32_t                i;
    int32_t             status = SystemP_SUCCESS;

    *errCode       = 0;
    
    if((initCfg == NULL) || (initCfg->hwaHandle == NULL))
    {
        *errCode = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }    

    // /* create heap for Doppler Hwa object. */
    // HeapP_construct(&gDopplerProcDDMAHeapObj, gDopplerProcDDMAHeapMem, DOPPLERPROCHWADDMA_HEAP_MEM_SIZE);
    
    // /* Allocate memory */
    // obj = HeapP_alloc(&gDopplerProcDDMAHeapObj, sizeof(DPU_DopplerProcHWA_Obj));
    // if(obj == NULL)
    // {
    //     *errCode = DPU_DOPPLERPROCHWA_ENOMEM;
    //     goto exit;
    // }
    obj = gDebugDopplerProcHwaObj;

    // obj = MemoryP_ctrlAlloc(sizeof(DPU_DopplerProcHWA_Obj), 0U);
    // if(obj == NULL)
    // {
    //     *errCode = DPU_DOPPLERPROCHWA_ENOMEM;
    //     goto exit;
    // }

    /* Initialize memory */
    memset((void *)obj, 0U, sizeof(DPU_DopplerProcHWA_Obj));
    
    /* Save init config params */
    obj->hwaHandle   = initCfg->hwaHandle;

    /* Creating semaphores */
    {{

    /* Create semaphore for HWA decompression done */
    status = SemaphoreP_constructBinary(&obj->decompHwaDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    } 

    /* Create semaphore for HWA decompression done */
    status = SemaphoreP_constructBinary(&obj->decompEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }       

    /* Create semaphore for EDMA done for Doppler FFT Ping Out */
    status = SemaphoreP_constructBinary(&obj->dopplerFFTPingEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }

    /* Create semaphore for EDMA done for DDMA Metric Ping Out */
    status = SemaphoreP_constructBinary(&obj->DDMAMetricPingEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }    

    /* Create semaphore for EDMA done for SumTx Ping Out */
    status = SemaphoreP_constructBinary(&obj->sumLogAbsPingEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }

    /* Create semaphore for EDMA done for Doppler FFT Pong Out */
    status = SemaphoreP_constructBinary(&obj->dopplerFFTPongEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }    

    /* Create semaphore for EDMA done for DDMA Metric Pong Out */
    status = SemaphoreP_constructBinary(&obj->DDMAMetricPongEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }

    /* Create semaphore for EDMA done for SumTx Pong Out */
    status = SemaphoreP_constructBinary(&obj->sumLogAbsPongEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }    

    /* Create semaphore for EDMA done for Azim FFT Ping Out */
    status = SemaphoreP_constructBinary(&obj->azimFFTPingEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }

    /* Create semaphore for EDMA done for CFAR Ping Out */
    status = SemaphoreP_constructBinary(&obj->cfarPingEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }    

    /* Create semaphore for EDMA done for LM Ping Out */
    status = SemaphoreP_constructBinary(&obj->localMaxPingEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }

    /* Create semaphore for EDMA done for Azim FFT Pong Out */
    status = SemaphoreP_constructBinary(&obj->azimFFTPongEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }    

    /* Create semaphore for EDMA done for CFAR Pong Out */
    status = SemaphoreP_constructBinary(&obj->cfarPongEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }

    /* Create semaphore for EDMA done for LM Pong Out */
    status = SemaphoreP_constructBinary(&obj->localMaxPongEdmaOutDoneSemaHandle, 0);
    if(status != SystemP_SUCCESS)
    {
        *errCode = DPU_DOPPLERPROCHWA_ESEMA;
        goto exit;
    }    

    }}


    
    /* Populate HWA base addresses and offsets. This is done only once, at init time.*/
    *errCode =  HWA_getHWAMemInfo(obj->hwaHandle, &hwaMemInfo);
    if (*errCode < 0)
    {       
        goto exit;
    }
    
    /* check if we have enough memory banks*/
    if(hwaMemInfo.numBanks < DPU_DOPPLERPROCHWA_NUM_HWA_MEMBANKS)
    {    
        *errCode = DPU_DOPPLERPROCHWA_EHWARES;
        goto exit;
    }
    
    for (i = 0; i < DPU_DOPPLERPROCHWA_NUM_HWA_MEMBANKS; i++)
    {
        obj->hwaMemBankAddr[i] = hwaMemInfo.baseAddress + i * hwaMemInfo.bankSize;
    }
 
exit:    

    if(*errCode < 0)
    {
        if(obj != NULL)
        {
            HeapP_free(&gDopplerProcDDMAHeapObj, obj);
            HeapP_destruct(&gDopplerProcDDMAHeapObj);
            // MemoryP_ctrlFree(obj, sizeof(DPU_DopplerProcHWA_Obj));
            obj = NULL;
        }
    }
   return ((DPU_DopplerProcHWA_Handle)obj);
}

/**
  *  @b Description
  *  @n
  *   Doppler DPU configuration 
  *
  *  @param[in]   handle     DPU handle.
  *  @param[in]   cfg        Pointer to configuration parameters.
  *
  *  \ingroup    DPU_DOPPLERPROC_EXTERNAL_FUNCTION
  *
  *  @retval
  *      Success      = 0
  *  @retval
  *      Error       != 0 @ref DPU_DOPPLERPROC_ERROR_CODE
  */
int32_t DPU_DopplerProcHWA_config
(
    DPU_DopplerProcHWA_Handle    handle,
    DPU_DopplerProcHWA_Config    *cfg
)
{
    DPU_DopplerProcHWA_Obj   *obj;
    int32_t                  retVal = 0;
    uint32_t idx;
    // uint16_t                 expectedWinSamples; //TODO
    int32_t                 scratchVal;

    obj = (DPU_DopplerProcHWA_Obj *)handle;
    if(obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    // CacheP_wbInv(cfg, sizeof(DPU_DopplerProcHWA_Config));
    //TODO Add check params
    obj->edmaHandle  = cfg->hwRes.edmaCfg.edmaHandle;
    //TODO
    obj->numDopplerChirps = cfg->staticCfg.numChirps;
    obj->numDopplerBins = mathUtils_getValidFFTSize(cfg->staticCfg.numChirps);

    /* Decompression params */
    {{
    obj->decompCfg.isEnabled = cfg->staticCfg.decompCfg.isEnabled;

    obj->decompCfg.compressionMethod = cfg->staticCfg.decompCfg.compressionMethod;
    obj->decompCfg.bytesPerSample = sizeof(cmplx16ImRe_t);
    obj->decompCfg.rxAntPerBlock = cfg->staticCfg.decompCfg.numRxAntennaPerBlock;
    obj->decompCfg.rangeBinsPerBlock = cfg->staticCfg.decompCfg.rangeBinsPerBlock;
    obj->decompCfg.outputSamplesPerBlock = obj->decompCfg.rxAntPerBlock * cfg->staticCfg.decompCfg.rangeBinsPerBlock; //OK
    obj->decompCfg.outputBytesPerBlock = obj->decompCfg.outputSamplesPerBlock * obj->decompCfg.bytesPerSample; //OK
    obj->decompCfg.numBlocks = cfg->staticCfg.numRangeBins * cfg->staticCfg.numRxAntennas / obj->decompCfg.outputSamplesPerBlock;
    obj->decompCfg.inputBytesPerBlock = (uint16_t)(((uint16_t) ((cfg->staticCfg.decompCfg.compressionRatio * 
                                    obj->decompCfg.outputBytesPerBlock + 3)/4)) * 4); /* Word aligned */ 
    obj->decompCfg.inputSamplesPerBlock = obj->decompCfg.inputBytesPerBlock / obj->decompCfg.bytesPerSample;
    obj->decompCfg.achievedCompressionRatio = (float) obj->decompCfg.inputBytesPerBlock / obj->decompCfg.outputBytesPerBlock;
    obj->decompCfg.decompEdmaToHwaStartAddress = (void *)cfg->hwRes.radarCube.data;
    obj->decompCfg.numChirpsPerPing = DECOMP_HWA_MEMBANK_SIZE / obj->decompCfg.outputBytesPerBlock; //TODO

    /* dstCIdx of DPU_DopplerProcHWA_configEdmaDecompressionOut must be less that 32768 */
    if(obj->decompCfg.numChirpsPerPing * obj->decompCfg.outputBytesPerBlock * 2 >= 32768){
        obj->decompCfg.numChirpsPerPing = obj->decompCfg.numChirpsPerPing/2;
    }

    obj->decompCfg.outerBlockSizeCompressed = obj->decompCfg.inputBytesPerBlock * cfg->staticCfg.numChirps;
    obj->decompCfg.numOuterBlocks = cfg->staticCfg.numRangeBins / cfg->staticCfg.decompCfg.rangeBinsPerBlock;

    /* numLoops is ceil(numChirps/numChirpsPerPing)/2 */
    if(cfg->staticCfg.numChirps % obj->decompCfg.numChirpsPerPing == 0){
        obj->decompCfg.numLoops = (cfg->staticCfg.numChirps / obj->decompCfg.numChirpsPerPing);
        if(obj->decompCfg.numLoops % 2 == 0){
            obj->decompCfg.numLoops = obj->decompCfg.numLoops/2;
        }
    }
    else{
        obj->decompCfg.numLoops = (cfg->staticCfg.numChirps / obj->decompCfg.numChirpsPerPing + 1);
        if(obj->decompCfg.numLoops % 2 == 0){
            obj->decompCfg.numLoops = obj->decompCfg.numLoops/2;
        }
    }
    if(obj->decompCfg.numLoops < 1){
        obj->decompCfg.numLoops = 2;
    }
    CacheP_wbInv(obj, sizeof(DPU_DopplerProcHWA_Obj), CacheP_TYPE_ALLD);
    obj->decompCfg.numChirpsPerPing = cfg->staticCfg.numChirps/(obj->decompCfg.numLoops * 2);
    if(obj->decompCfg.numChirpsPerPing < 1){
        retVal = DPU_DOPPLERPROCHWA_ERROR_NUMCHIRPSPERPING;
        goto exit;
    }
    obj->decompCfg.numBlocksPerPing = obj->decompCfg.numChirpsPerPing;

    if(obj->decompCfg.rxAntPerBlock != cfg->staticCfg.numRxAntennas){
        retVal = DPU_DOPPLERPROCHWA_ERROR_NUMRXANTPERBLOCK_DECOMPRESSION;
        goto exit;
    }

    /* RangeBinsPerBlock should be a power of 2 */
    if(obj->decompCfg.rangeBinsPerBlock == 0 || 
        ((obj->decompCfg.rangeBinsPerBlock & (obj->decompCfg.rangeBinsPerBlock - 1)) != 0)){
        retVal = DPU_DOPPLERPROCHWA_ERROR_RANGEBINSPERBLOCK_DECOMPRESSION;
        goto exit;
    }

    if(obj->decompCfg.compressionMethod != 0){
        retVal = DPU_DOPPLERPROCHWA_ERROR_METHOD_DECOMPRESSION;
        goto exit;
    }

    /* Populate HWA Common config structure */
    obj->decompCfg.hwaCommonConfig.configMask = HWA_COMMONCONFIG_MASK_STATEMACHINE_CFG |/* numLoops, paramStartIdx, paramStopIdx combined here */
                               HWA_COMMONCONFIG_MASK_EGECOMRESS_KPARAM;

    obj->decompCfg.hwaCommonConfig.numLoops = obj->decompCfg.numLoops;
    obj->decompCfg.hwaCommonConfig.paramStartIdx = cfg->hwRes.hwaCfg.decompStageHwaStateMachineCfg.paramSetStartIdx; 
    obj->decompCfg.hwaCommonConfig.paramStopIdx = cfg->hwRes.hwaCfg.decompStageHwaStateMachineCfg.paramSetStartIdx + cfg->hwRes.hwaCfg.decompStageHwaStateMachineCfg.numParamSets - 1U;

    //TODO
    obj->decompCfg.hwaDmaTriggerSourcePingPongIn[0] = obj->decompCfg.hwaCommonConfig.paramStartIdx + DECOMP_PING_HWA_PARAMSET_RELATIVE_IDX;
    obj->decompCfg.hwaDmaTriggerSourcePingPongIn[1] = obj->decompCfg.hwaCommonConfig.paramStartIdx + DECOMP_PONG_HWA_PARAMSET_RELATIVE_IDX;
    // obj->decompCfg.hwaDmaTriggerSourcePingPongOut[0] = obj->hwaParamStartIdx + DECOMP_PING_HWA_PARAMSET_RELATIVE_IDX - 1; //Dummy ping //TODO
    // obj->decompCfg.hwaDmaTriggerSourcePingPongOut[1] = obj->hwaParamStartIdx + DECOMP_PONG_HWA_PARAMSET_RELATIVE_IDX - 1; //Dummy pong //TODO


    /* EGE Compression values */
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[0] = 3;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[1] = 4;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[2] = 5;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[3] = 7;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[4] = 9;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[5] = 11;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[6] = 13;
    obj->decompCfg.hwaCommonConfig.compressConfig.EGEKparam[7] = 15;

    // CSL_FINSR(((DSSHWACCRegs* )0x06062000)->SINGLE_STEP_EN, SINGLE_STEP_EN_SINGLE_STEP_EN_END, SINGLE_STEP_EN_SINGLE_STEP_EN_START, 1U);
    // CSL_FINSR(((DSSHWACCRegs* )0x06062000)->SINGLE_STEP_TRIG, SINGLE_STEP_TRIG_SINGLE_STEP_TRIG_END, SINGLE_STEP_TRIG_SINGLE_STEP_TRIG_START, 1U);

    /**********************************************/
    /* ENABLE NUMLOOPS DONE INTERRUPT FROM HWA    */
    /**********************************************/
    retVal = HWA_enableDoneInterrupt(obj->hwaHandle,
									 HWA_THREAD_BACKGROUNDCONTEXT,
                                     DPU_DopplerProcHWA_hwaDoneIsrCallback,
                                     (void *)&obj->decompHwaDoneSemaHandle);

    retVal = DPU_DopplerProcHWA_configHwaDecompression(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configEdmaDecompressionIn(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configEdmaDecompressionOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    }}

    /* Doppler demod params */
    {{
    obj->dopplerDemodCfg.numBandsActive = cfg->staticCfg.numTxAntennas;
    
    /* Empty subbands */
    switch (cfg->staticCfg.numTxAntennas)
    {
        case 2: 
            obj->dopplerDemodCfg.numBandsEmpty = 1;
            break;
        case 3:
            obj->dopplerDemodCfg.numBandsEmpty = 1;
            break;
        case 4: 
            obj->dopplerDemodCfg.numBandsEmpty = 2;
            break;
        default:
            retVal = DPU_DOPPLERPROCHWA_EINVAL;
            goto exit;
    }

    obj->dopplerDemodCfg.numBandsTotal = obj->dopplerDemodCfg.numBandsActive + obj->dopplerDemodCfg.numBandsEmpty;
    if (obj->dopplerDemodCfg.numBandsTotal != cfg->staticCfg.numBandsTotal){
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    obj->dopplerDemodCfg.dopplerIOCfg.input.isReal = 0;
    obj->dopplerDemodCfg.dopplerIOCfg.input.bytesPerSample = sizeof(cmplx16ImRe_t);
    obj->dopplerDemodCfg.dopplerIOCfg.input.isSigned = 1;
    obj->dopplerDemodCfg.dopplerIOCfg.output.isReal = 0;
    obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample = sizeof(cmplx32ImRe_t);
    obj->dopplerDemodCfg.dopplerIOCfg.output.isSigned = 1;

    obj->dopplerDemodCfg.logAbsIOCfg.input.isReal = obj->dopplerDemodCfg.dopplerIOCfg.output.isReal;
    obj->dopplerDemodCfg.logAbsIOCfg.input.bytesPerSample = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    obj->dopplerDemodCfg.logAbsIOCfg.input.isSigned = obj->dopplerDemodCfg.dopplerIOCfg.output.isSigned;
    obj->dopplerDemodCfg.logAbsIOCfg.output.isReal = 1;
    obj->dopplerDemodCfg.logAbsIOCfg.output.bytesPerSample = sizeof(uint16_t);
    obj->dopplerDemodCfg.logAbsIOCfg.output.isSigned = 0;

    obj->dopplerDemodCfg.sumRxIOCfg.input.isReal = obj->dopplerDemodCfg.logAbsIOCfg.output.isReal;
    obj->dopplerDemodCfg.sumRxIOCfg.input.bytesPerSample = obj->dopplerDemodCfg.logAbsIOCfg.output.bytesPerSample;
    obj->dopplerDemodCfg.sumRxIOCfg.input.isSigned = obj->dopplerDemodCfg.logAbsIOCfg.output.isSigned;
    obj->dopplerDemodCfg.sumRxIOCfg.output.isReal = 1;
    obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample = sizeof(uint16_t);
    obj->dopplerDemodCfg.sumRxIOCfg.output.isSigned = 0;

    obj->dopplerDemodCfg.DDMAMetricIOCfg.input.isReal = obj->dopplerDemodCfg.sumRxIOCfg.output.isReal;
    obj->dopplerDemodCfg.DDMAMetricIOCfg.input.bytesPerSample = obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample;
    obj->dopplerDemodCfg.DDMAMetricIOCfg.input.isSigned = obj->dopplerDemodCfg.sumRxIOCfg.output.isSigned;
    obj->dopplerDemodCfg.DDMAMetricIOCfg.output.isReal = 1;
    obj->dopplerDemodCfg.DDMAMetricIOCfg.output.bytesPerSample = sizeof(uint32_t);
    obj->dopplerDemodCfg.DDMAMetricIOCfg.output.isSigned = 0;

    obj->dopplerDemodCfg.sumTxIOCfg.input.isReal = obj->dopplerDemodCfg.sumRxIOCfg.output.isReal;
    obj->dopplerDemodCfg.sumTxIOCfg.input.bytesPerSample = obj->dopplerDemodCfg.sumRxIOCfg.output.bytesPerSample;
    obj->dopplerDemodCfg.sumTxIOCfg.input.isSigned = obj->dopplerDemodCfg.sumRxIOCfg.output.isSigned;
    obj->dopplerDemodCfg.sumTxIOCfg.output.isReal = 1;
    obj->dopplerDemodCfg.sumTxIOCfg.output.bytesPerSample = sizeof(uint16_t);
    obj->dopplerDemodCfg.sumTxIOCfg.output.isSigned = 0;

    obj->dopplerDemodCfg.hwaDmaTriggerSourcePingPongIn[0] = cfg->hwRes.hwaCfg.dopplerStageHwaStateMachineCfg.paramSetStartIdx + DPU_DOPPLERHWADDMA_DOPPLER_FFT_PING_HWA_PARAMSET_RELATIVE_IDX - 1; //TODO -1 should go
    obj->dopplerDemodCfg.hwaDmaTriggerSourcePingPongIn[1] = cfg->hwRes.hwaCfg.dopplerStageHwaStateMachineCfg.paramSetStartIdx + DPU_DOPPLERHWADDMA_DOPPLER_FFT_PONG_HWA_PARAMSET_RELATIVE_IDX - 1; //TODO

    /* Config Common Registers */
    obj->dopplerDemodCfg.hwaCommonConfig.configMask = HWA_COMMONCONFIG_MASK_STATEMACHINE_CFG |/* numLoops, paramStartIdx, paramStopIdx combined here */
                                                        HWA_COMMONCONFIG_MASK_TWIDDITHERENABLE |
                                                        HWA_COMMONCONFIG_MASK_LFSRSEED;

    obj->dopplerDemodCfg.hwaCommonConfig.fftConfig.twidDitherEnable = HWA_FEATURE_BIT_ENABLE;
    obj->dopplerDemodCfg.hwaCommonConfig.fftConfig.lfsrSeed = 0x1234567; /*Some non-zero value*/
    obj->dopplerDemodCfg.hwaCommonConfig.numLoops      = cfg->staticCfg.decompCfg.rangeBinsPerBlock / 2;  
    obj->dopplerDemodCfg.hwaCommonConfig.paramStartIdx = cfg->hwRes.hwaCfg.dopplerStageHwaStateMachineCfg.paramSetStartIdx; 
    obj->dopplerDemodCfg.hwaCommonConfig.paramStopIdx  = cfg->hwRes.hwaCfg.dopplerStageHwaStateMachineCfg.paramSetStartIdx + cfg->hwRes.hwaCfg.dopplerStageHwaStateMachineCfg.numParamSets - 1U; 
    
    /* Populate Shuffle LUT RAM contents in array */
    for(idx = 0; idx < obj->dopplerDemodCfg.numBandsTotal; idx++){
        cfg->hwRes.shuffleRAM[idx] = idx * (obj->numDopplerBins/obj->dopplerDemodCfg.numBandsTotal);
    }

    /* Populate Shuffle LUT RAM in HWA */
    retVal = HWA_configRam(obj->hwaHandle, HWA_RAM_TYPE_SHUFFLE_RAM, (uint8_t *)&cfg->hwRes.shuffleRAM[0], sizeof(cfg->hwRes.shuffleRAM), 0); //TODO SHUFFLERAM MAY NOT BE MAXSIZE
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configHwaDopplerFFTDDMADemod(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configEdmaDopplerIn(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configEdmaDopplerFFTOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configEdmaDDMAMetricOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    retVal = DPU_DopplerProcHWA_configEdmaDopplerFFTSumTxOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    
    }}

    /* Azimuth FFT - CFAR params */
    {{
    
    obj->cfarAzimFFTCfg.numAzimFFTBins = 4 * mathUtils_getValidFFTSize(cfg->staticCfg.numRxAntennas * cfg->staticCfg.numAzimTxAntennas);
   
    obj->cfarAzimFFTCfg.azimFFTIOCfg.input.isReal = 0;
    obj->cfarAzimFFTCfg.azimFFTIOCfg.input.bytesPerSample = obj->dopplerDemodCfg.dopplerIOCfg.output.bytesPerSample;
    obj->cfarAzimFFTCfg.azimFFTIOCfg.input.isSigned = obj->dopplerDemodCfg.dopplerIOCfg.output.isSigned;
    obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isReal = 1;
    obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample = sizeof(uint16_t);
    obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isSigned = 0;

    obj->cfarAzimFFTCfg.cfarIOCfg.input.isReal = 1;
    obj->cfarAzimFFTCfg.cfarIOCfg.input.bytesPerSample = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample;
    obj->cfarAzimFFTCfg.cfarIOCfg.input.isSigned = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isSigned;
    obj->cfarAzimFFTCfg.cfarIOCfg.output.isReal = 0;
    obj->cfarAzimFFTCfg.cfarIOCfg.output.bytesPerSample = sizeof(cmplx32ImRe_t);
    obj->cfarAzimFFTCfg.cfarIOCfg.output.isSigned = 0;

    obj->cfarAzimFFTCfg.localMaxIOCfg.input.isReal = 1;
    obj->cfarAzimFFTCfg.localMaxIOCfg.input.bytesPerSample = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.bytesPerSample;
    obj->cfarAzimFFTCfg.localMaxIOCfg.input.isSigned = obj->cfarAzimFFTCfg.azimFFTIOCfg.output.isSigned;
    obj->cfarAzimFFTCfg.localMaxIOCfg.output.isReal = 1;
    obj->cfarAzimFFTCfg.localMaxIOCfg.output.bytesPerSample = sizeof(uint32_t);
    obj->cfarAzimFFTCfg.localMaxIOCfg.output.isSigned = 0;

    obj->cfarAzimFFTCfg.cfarCfg.averageMode = cfg->staticCfg.cfarCfg.averageMode; /* CFAR_OS */
    if(obj->cfarAzimFFTCfg.cfarCfg.averageMode != 3){ /* CFAR_OS */
        retVal = DPU_DOPPLERPROCHWA_ERROR_METHOD_CFAR;
        goto exit;
    }
    obj->cfarAzimFFTCfg.cfarCfg.winLen = cfg->staticCfg.cfarCfg.winLen; 
    obj->cfarAzimFFTCfg.cfarCfg.noiseDivShift = cfg->staticCfg.cfarCfg.noiseDivShift; 
    obj->cfarAzimFFTCfg.cfarCfg.guardLen = cfg->staticCfg.cfarCfg.guardLen;  /* Not applicable for CFAR-OS */
    if(obj->cfarAzimFFTCfg.cfarCfg.guardLen != 0){ /* CFAR_OS */
        retVal = DPU_DOPPLERPROCHWA_ERROR_METHOD_CFAR;
        goto exit;
    }
    obj->cfarAzimFFTCfg.cfarCfg.cyclicMode = cfg->staticCfg.cfarCfg.cyclicMode;
    obj->cfarAzimFFTCfg.cfarCfg.peakGroupingScheme = cfg->staticCfg.cfarCfg.peakGroupingScheme;
    obj->cfarAzimFFTCfg.cfarCfg.peakGroupingEn = cfg->staticCfg.cfarCfg.peakGroupingEn; 
    obj->cfarAzimFFTCfg.cfarCfg.osKvalue = cfg->staticCfg.cfarCfg.osKvalue; 
    obj->cfarAzimFFTCfg.cfarCfg.osEdgeKscaleEn = cfg->staticCfg.cfarCfg.osEdgeKscaleEn; 
    obj->cfarAzimFFTCfg.cfarCfg.thresholdScale = cfg->staticCfg.cfarCfg.thresholdScale; 

    obj->cfarAzimFFTCfg.localMaxCfg.azimThreshold = cfg->staticCfg.localMaxCfg.azimThreshold; 
    obj->cfarAzimFFTCfg.localMaxCfg.dopplerThreshold = cfg->staticCfg.localMaxCfg.dopplerThreshold; 

    obj->cfarAzimFFTCfg.hwaDmaTriggerSourcePingPongIn[DPU_DOPPLERPROCHWA_PING] = cfg->hwRes.hwaCfg.azimCfarStageHwaStateMachineCfg.paramSetStartIdx + DPU_DOPPLERHWADDMA_AZIMFFT_PING_HWA_PARAMSET_RELATIVE_IDX - 1;
    obj->cfarAzimFFTCfg.hwaDmaTriggerSourcePingPongIn[DPU_DOPPLERPROCHWA_PONG] = cfg->hwRes.hwaCfg.azimCfarStageHwaStateMachineCfg.paramSetStartIdx + DPU_DOPPLERHWADDMA_AZIMFFT_PONG_HWA_PARAMSET_RELATIVE_IDX - 1; //TODO: Remove the dummy stage?

    /* Config Common Registers */
    obj->cfarAzimFFTCfg.hwaCommonConfig.configMask = HWA_COMMONCONFIG_MASK_STATEMACHINE_CFG |/* numLoops, paramStartIdx, paramStopIdx combined here */
                                                    HWA_COMMONCONFIG_MASK_CFARTHRESHOLDSCALE |
                                                    HWA_COMMONCONFIG_MASK_MAX2D_OFFSETBOTHDIM |
                                                    HWA_COMMONCONFIG_MASK_COMPLEXMULT_SCALEARRAY;

    /* 2D maximum value offset */
    scratchVal = round(pow(10.0, (double)(obj->cfarAzimFFTCfg.localMaxCfg.azimThreshold)/20) * (1 << 10));
    obj->cfarAzimFFTCfg.hwaCommonConfig.advStatConfig.max2DoffsetDim1 = -scratchVal;
    scratchVal = round(pow(10.0, (double)(obj->cfarAzimFFTCfg.localMaxCfg.dopplerThreshold)/20) * (1 << 10));
    obj->cfarAzimFFTCfg.hwaCommonConfig.advStatConfig.max2DoffsetDim2 = -scratchVal;

    obj->cfarAzimFFTCfg.hwaCommonConfig.numLoops = cfg->staticCfg.decompCfg.rangeBinsPerBlock;
    obj->cfarAzimFFTCfg.hwaCommonConfig.paramStartIdx = cfg->hwRes.hwaCfg.azimCfarStageHwaStateMachineCfg.paramSetStartIdx;
    obj->cfarAzimFFTCfg.hwaCommonConfig.paramStopIdx =  cfg->hwRes.hwaCfg.azimCfarStageHwaStateMachineCfg.paramSetStartIdx + cfg->hwRes.hwaCfg.azimCfarStageHwaStateMachineCfg.numParamSets - 1U; ;
    obj->cfarAzimFFTCfg.hwaCommonConfig.cfarConfig.thresholdScale = obj->cfarAzimFFTCfg.cfarCfg.thresholdScale;

    /* Evaluate the float calb params into quantized values acceptable to HWA */
    retVal = mathUtils_asymQuantInt(cfg->staticCfg.antennaCalibParams, 
                                    (void *)obj->cfarAzimFFTCfg.antennaCalibParamsQuantized,
                                    cfg->staticCfg.numTxAntennas * cfg->staticCfg.numRxAntennas * 2, //TODO: Change to maxnum?
                                    1,
                                    20,
                                    1); /* Signed */
    if (retVal != 0)
    {
        goto exit;
    }

    for (idx = 0; idx < 24; idx+=2)
    /* vector multiplication vector */
    {
        obj->cfarAzimFFTCfg.hwaCommonConfig.complexMultiplyConfig.Iscale[idx/2] = obj->cfarAzimFFTCfg.antennaCalibParamsQuantized[idx+1]; /* Q20 format */
        obj->cfarAzimFFTCfg.hwaCommonConfig.complexMultiplyConfig.Qscale[idx/2] = obj->cfarAzimFFTCfg.antennaCalibParamsQuantized[idx];
    }

    /* Configure HWA paramsets */
    retVal = DPU_DopplerProcHWA_configHwaCFARAzimFFT(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    /* Configure EDMA for sending Doppler FFT data as input to HWA */
    retVal = DPU_DopplerProcHWA_configEdmaAzimFFTIn(obj, cfg, (uint32_t)(cfg->hwRes.dopFFTSubMat));
    if (retVal != 0)
    {
        goto exit;
    }

    /* Configure EDMA to bring Azimuth FFT data out of HWA */
    retVal = DPU_DopplerProcHWA_configEdmaAzimFFTOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    /* Configure EDMA to bring CFAR data out of HWA */
    retVal = DPU_DopplerProcHWA_configEdmaCfarOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }

    /* Configure EDMA to bring LM data out of HWA */
    retVal = DPU_DopplerProcHWA_configEdmaLocalMaxOut(obj, cfg);
    if (retVal != 0)
    {
        goto exit;
    }
    
    }}

    /* Validate params */
    if(!cfg ||
       !cfg->hwRes.edmaCfg.edmaHandle ||
       !cfg->hwRes.hwaCfg.window ||
       !cfg->hwRes.radarCube.data ||
       !cfg->hwRes.detMatrix.data
      )
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }

    /* Check if detection matrix size is sufficient*/
    if(cfg->hwRes.detMatrix.dataSize < (cfg->staticCfg.numRangeBins *
                                        obj->numDopplerBins * sizeof(uint16_t)))
    {
        retVal = DPU_DOPPLERPROCHWA_EDETMSIZE;
        goto exit;
    }

    //TODO: Add Proper Debug Checks

    // /* Check if radar cube format is supported by DPU*/ //TODO
    // if(cfg->hwRes.radarCube.datafmt != DPIF_RADARCUBE_FORMAT_2)
    // {
    //     retVal = DPU_DOPPLERPROCHWA_ECUBEFORMAT;
    //     goto exit;
    // }

    //TODO Add debug checks

    /* Even though Window RAM is not used by the first stage, there's no issue
    in programming it at this point itself */
    /* HWA window configuration */
    retVal = HWA_configRam(obj->hwaHandle,
                           HWA_RAM_TYPE_WINDOW_RAM,
                           (uint8_t *)cfg->hwRes.hwaCfg.window,
                           cfg->hwRes.hwaCfg.windowSize, //size in bytes
                           cfg->hwRes.hwaCfg.winRamOffset * sizeof(int32_t)); 
    if (retVal != 0)
    {
        goto exit;
    }

    /* Enable the HWA */
    retVal = HWA_enable(obj->hwaHandle, 1);
    if (retVal != 0)
    {
        goto exit;
    }


exit:
    return retVal;
}

 /**
  *  @b Description
  *  @n Doppler DPU process function. 
  *   
  *  @param[in]   handle     DPU handle.
  *  @param[in]   cfg        DPU config.
  *  @param[out]  outParams  Output parameters.
  *
  *  \ingroup    DPU_DOPPLERPROC_EXTERNAL_FUNCTION
  *
  *  @retval
  *      Success     =0
  *  @retval
  *      Error      !=0 @ref DPU_DOPPLERPROC_ERROR_CODE
  */
int32_t DPU_DopplerProcHWA_process
(
    DPU_DopplerProcHWA_Handle    handle,
    DPU_DopplerProcHWA_Config    *cfg,
    DPU_DopplerProcHWA_OutParams *outParams
)
{

    volatile uint32_t   startTime;
    DPU_DopplerProcHWA_Obj *obj;
    int32_t retVal;
    bool                status;
    uint32_t rangeBinIdx;
    uint32_t blockIdx = 0;

    obj = (DPU_DopplerProcHWA_Obj *)handle;
    if (obj == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
        goto exit;
    }
    /* Set inProgress state */
    obj->inProgress = true;
    
    obj->numObjOut = 0;
    
    startTime = CycleCounterP_getCount32(); //CycleprofilerP_getTimeStamp();

    /* Run block-wise */
    for(blockIdx = 0; blockIdx < obj->decompCfg.numOuterBlocks; blockIdx++){

        /************************************************/
        /* STAGE I (DECOMPRESSION)                      */
        /************************************************/

        /* Disable the HWA */
        retVal = HWA_enable(obj->hwaHandle,0); 
        if (retVal != 0)
        {
            goto exit;
        }

        /* Config Common Registers for decompression stage */
        retVal = HWA_configCommon(obj->hwaHandle, &obj->decompCfg.hwaCommonConfig);
        if (retVal != 0)
        {
            goto exit;
        }

        /* Enable the HWA */
        retVal = HWA_enable(obj->hwaHandle,1); 
        if (retVal != 0)
        {
            goto exit;
        }

        /* Update source address for decompression */
        if(blockIdx != 0){
            
            obj->decompCfg.decompEdmaToHwaStartAddress = (int32_t *)((uint8_t *)obj->decompCfg.decompEdmaToHwaStartAddress + 
                                                                    obj->decompCfg.outerBlockSizeCompressed);
            
            // retVal = EDMA_setSourceAddress(obj->edmaHandle,
            //                     cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[0].channel,
            //                     (uint32_t)obj->decompCfg.decompEdmaToHwaStartAddress);
            retVal = DPEDMA_updateAddressAndTrigger(obj->edmaHandle,
                                (uint32_t)obj->decompCfg.decompEdmaToHwaStartAddress, /* src addr */
                                NULL,                                                 /* don't update dest addr */
                                cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[0].channel, /* param Id */
                                false);                                               /* don't trigger channel */
            if (retVal != 0)
            {
                goto exit;
            }

            // retVal = EDMA_setSourceAddress(obj->edmaHandle,
            //                     cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[1].channel,
            //                     (uint32_t)((uint8_t *)obj->decompCfg.decompEdmaToHwaStartAddress 
            //                         + obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing)); 
            retVal = DPEDMA_updateAddressAndTrigger(obj->edmaHandle,
                                (uint32_t)obj->decompCfg.decompEdmaToHwaStartAddress
                                        + obj->decompCfg.inputBytesPerBlock * obj->decompCfg.numChirpsPerPing, /* src addr */
                                NULL,                                                 /* don't update dest addr */
                                cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[1].channel, /* param Id */
                                false);                                               /* don't trigger channel */
            if (retVal != 0)
            {
                goto exit;
            }
            
        }

        /* Start ping DMA Transfer for decompression */
        retVal = DPEDMA_edmaStartTransferManualTrigger(obj->edmaHandle, (uint32_t)cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[0].channel);
        if (retVal != 0)
        {
            goto exit;
        }
        /* Start pong DMA Transfer for decompression */
        retVal = DPEDMA_edmaStartTransferManualTrigger(obj->edmaHandle, (uint32_t)cfg->hwRes.edmaCfg.decompEdmaCfg.edmaIn.pingPong[1].channel);
        if (retVal != 0)
        {
            goto exit;
        }

        /**********************************************/
        /* WAIT FOR HWA/EDMA DECOMP NUMLOOPS INTERRUPT     */
        /**********************************************/
        status = SemaphoreP_pend(&obj->decompHwaDoneSemaHandle, SystemP_WAIT_FOREVER);
        if (status != SystemP_SUCCESS)
        {
            retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
            goto exit;
        }
        status = SemaphoreP_pend(&obj->decompEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
        if (status != SystemP_SUCCESS)
        {
            retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
            goto exit;
        }

        /************************************************/
        /* STAGE II (DOPPLER FFT, DEMOD, AZIM, OBJLIST) */
        /************************************************/

        for(rangeBinIdx = 0; rangeBinIdx < cfg->staticCfg.decompCfg.rangeBinsPerBlock; rangeBinIdx+=2){

            /************************************************/
            /* HWA COMMON CONFIG FOR DOPPLERFFT/DEMOD STAGE */
            /************************************************/
            
            /* Disable the HWA */
            retVal = HWA_enable(obj->hwaHandle,0); 
            if (retVal != 0)
            {
                goto exit;
            }

            /* Config Common Registers */
            retVal = HWA_configCommon(obj->hwaHandle, &obj->dopplerDemodCfg.hwaCommonConfig);
            if (retVal != 0)
            {
                goto exit;
            }

            /* Enable the HWA */
            retVal = HWA_enable(obj->hwaHandle,1); 
            if (retVal != 0)
            {
                goto exit;
            }

            /* Send out decompressed range bin to HWA for Doppler FFT calculation (ping) */
            retVal = DPEDMA_edmaStartTransferManualTrigger(obj->edmaHandle, (uint32_t)cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIn.pingPong[DPU_DOPPLERPROCHWA_PING].channel);
            if (retVal != 0)
            {
                goto exit;
            }

            /* Extract object list of the previous pong stage */
            if(rangeBinIdx != 0){

                retVal = DPU_DopplerProcHWA_extractObjectList(obj, cfg, blockIdx, rangeBinIdx - 1);
                if (retVal != 0)
                {
                    goto exit;
                }

            }

            /* Send out decompressed range bin to HWA for Doppler FFT calculation (pong) */
            retVal = DPEDMA_edmaStartTransferManualTrigger(obj->edmaHandle, (uint32_t)cfg->hwRes.edmaCfg.dopplerEdmaCfg.edmaIn.pingPong[DPU_DOPPLERPROCHWA_PONG].channel);
            if (retVal != 0)
            {
                goto exit;
            }
            
            /* Wait for Doppler FFT ping data transfer */
            status = SemaphoreP_pend(&obj->dopplerFFTPingEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Wait for DDMA Metric ping data transfer */
            status = SemaphoreP_pend(&obj->DDMAMetricPingEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* If we do an immediate trigger, EDMA (DDMA Metric Out) and HWA (Sum Tx Out) will
               try to access the same HWA MemBank (M0) which was seen to cause issues */
            HWA_setSoftwareTrigger(obj->hwaHandle);

            /* Wait for SumLogAbs (Sum Tx) ping transfer */
            status = SemaphoreP_pend(&obj->sumLogAbsPingEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Ping Demodulation */
            DPU_DopplerProcHWA_DDMADemod(obj, cfg, 0, 0);

            /* Invalidate and write back the ping doppler FFT sub matrix */
            CacheP_wbInv((void *)cfg->hwRes.dopFFTSubMat, cfg->hwRes.dopFFTSubMatSizeBytes/2, CacheP_TYPE_ALLD);
            CacheP_wbInv((void *)cfg->hwRes.dopMaxSubBandScratchBuf[0],
                         cfg->hwRes.dopMaxSubBandScratchBufferSizeBytes/2, CacheP_TYPE_ALLD);

            /* Wait for Doppler FFT pong data transfer */
            status = SemaphoreP_pend(&obj->dopplerFFTPongEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Wait for DDMA Metric pong data transfer */
            status = SemaphoreP_pend(&obj->DDMAMetricPongEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }
            
            /* If we do an immediate trigger, EDMA (DDMA Metric Out) and HWA (Sum Tx Out) will
               try to access the same HWA MemBank (M0) which was seen to cause issues */
            HWA_setSoftwareTrigger(obj->hwaHandle);

            /* Wait for SumLogAbs (Sum Tx) ping transfer */
            status = SemaphoreP_pend(&obj->sumLogAbsPongEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Disable the HWA */
            retVal = HWA_enable(obj->hwaHandle,0); 
            if (retVal != 0)
            {
                goto exit;
            }
            /* Config Common Registers for Azim-CFAR */ //TODO: This can be done in the doppler stage itself?
            retVal = HWA_configCommon(obj->hwaHandle, &obj->cfarAzimFFTCfg.hwaCommonConfig);
            if (retVal != 0)
            {
                goto exit;
            }

            /* Disable the HWA */
            retVal = HWA_enable(obj->hwaHandle,1); 
            if (retVal != 0)
            {
                goto exit;
            }

            /* Start ping transfer of Doppler FFT data into HWA for Azim FFT calculation */
            retVal = DPEDMA_edmaStartTransferManualTrigger(obj->edmaHandle, (uint32_t)cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTIn.pingPong[DPU_DOPPLERPROCHWA_PING].channel);
            if (retVal != 0)
            {
                goto exit;
            }

            /* Pong Demodulation */
            DPU_DopplerProcHWA_DDMADemod(obj, cfg, 0, 1);

            /* Invalidate and write back the cache for pong doppler FFT sub matrix */
            CacheP_wbInv((void *)((uint8_t *)cfg->hwRes.dopFFTSubMat + cfg->hwRes.dopFFTSubMatSizeBytes/2),
                         cfg->hwRes.dopFFTSubMatSizeBytes/2, CacheP_TYPE_ALLD);
            CacheP_wbInv((void *)cfg->hwRes.dopMaxSubBandScratchBuf[1],
                         cfg->hwRes.dopMaxSubBandScratchBufferSizeBytes/2, CacheP_TYPE_ALLD);

            /* Wait for CFAR ping data transfer out from HWA */
            status = SemaphoreP_pend(&obj->cfarPingEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Wait for Local Max ping data transfer out from HWA */
            status = SemaphoreP_pend(&obj->localMaxPingEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Wait for Azim FFT ping data transfer out from HWA */
            status = SemaphoreP_pend(&obj->azimFFTPingEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }
            
            /* Read ping CFAR peak reg count in HWA */
            retVal = HWA_readCFARPeakCountReg(obj->hwaHandle, (uint8_t *)&obj->numCfarPeaksPing, sizeof(uint32_t));
            if (retVal != 0)
            {
                goto exit;
            }

            /* This EDMA transfer is done later here since we need to read the CFAR peak count reg which cannot
            be done while another paramset is executing. Hence we cannot have any pong paramset running while we
            read the CFAR peak count register. The pong paramset can run while the DSP is creating the object list
            at the ping side, hence the EDMA transferred is triggered next here. */
            retVal = DPEDMA_edmaStartTransferManualTrigger(obj->edmaHandle, (uint32_t)cfg->hwRes.edmaCfg.azimCfarEdmaCfg.edmaAzimFFTIn.pingPong[DPU_DOPPLERPROCHWA_PONG].channel);
            if (retVal != 0)
            {
                goto exit;
            }

            /* Extract ping object list */
            retVal = DPU_DopplerProcHWA_extractObjectList(obj, cfg, blockIdx, rangeBinIdx);
            if (retVal != 0)
            {
                goto exit;
            }

            /* Wait for Azim FFT pong data transfer out from HWA */
            status = SemaphoreP_pend(&obj->azimFFTPongEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Wait for CFAR pong data transfer out from HWA */
            status = SemaphoreP_pend(&obj->cfarPongEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Wait for Local Max pong data transfer out from HWA */
            status = SemaphoreP_pend(&obj->localMaxPongEdmaOutDoneSemaHandle, SystemP_WAIT_FOREVER);
            if (status != SystemP_SUCCESS)
            {
                retVal = DPU_DOPPLERPROCHWA_ESEMASTATUS;
                goto exit;
            }

            /* Read pong CFAR peak reg count in HWA */
            retVal = HWA_readCFARPeakCountReg(obj->hwaHandle, (uint8_t *)&obj->numCfarPeaksPong, sizeof(uint32_t));
            if (retVal != 0)
            {
                goto exit;
            }
        }

        /* Extract object list of the last pong stage */
        if(rangeBinIdx != 0){

            retVal = DPU_DopplerProcHWA_extractObjectList(obj, cfg, blockIdx, rangeBinIdx - 1);
            if (retVal != 0)
            {
                goto exit;
            }

        }
    }

    outParams->numObjOut = obj->numObjOut;
    
    outParams->stats.numProcess++;
    outParams->stats.processingTime = CycleCounterP_getCount32() - startTime; //CycleprofilerP_getTimeStamp() - startTime;
    
exit:
    if (obj != NULL)
    {
        obj->inProgress = false;
    }    
    
    return retVal;
}

/**
  *  @b Description
  *  @n
  *  Doppler DPU deinit 
  *
  *  @param[in]   handle   DPU handle.
  *
  *  \ingroup    DPU_DOPPLERPROC_EXTERNAL_FUNCTION
  *
  *  @retval
  *      Success      =0
  *  @retval
  *      Error       !=0 @ref DPU_DOPPLERPROC_ERROR_CODE
  */
int32_t DPU_DopplerProcHWA_deinit(DPU_DopplerProcHWA_Handle handle)
{
    int32_t     retVal = 0;
    
    /* Free memory */
    if(handle == NULL)
    {
        retVal = DPU_DOPPLERPROCHWA_EINVAL;
    }
    else
    {
        // MemoryP_ctrlFree(handle, sizeof(DPU_DopplerProcHWA_Obj));
        HeapP_free(&gDopplerProcDDMAHeapObj, handle);
        HeapP_destruct(&gDopplerProcDDMAHeapObj);
    }
    
    return retVal;
}

