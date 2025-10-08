/**
 *   @file  dopplerprochwainternal.h
 *
 *   @brief
 *      Implements Data path doppler processing functionality.
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
#ifndef DOPPLERPROC_HWA_INTERNAL_H
#define DOPPLERPROC_HWA_INTERNAL_H

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* PDK files */
// #include <ti/drv/hwa/hwa.h>
#include <drivers/hwa.h>
#include <kernel/dpl/CycleCounterP.h>
// #include <osal/CycleprofilerP.h>

/* DPIF Components Include Files */
#include <ti/datapath/dpif/dpif_detmatrix.h>
#include <ti/datapath/dpif/dpif_radarcube.h>

/* mmWave SDK Data Path Include Files */
#include <ti/datapath/dpif/dp_error.h>
#include "../dopplerprochwaDDMA.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 *  DC estimation parameters for HWA
 *
 * @details
 *  The structure holds DC estimation parameters to configure HWA.
 *  The HWA DC estimation consists of the following:
 *  - accumulate of input samples
 *  - multiply accumulated values by dcestScale, and right shift by (8 + 6 + dcestShift)
 *  
 *
 *  \ingroup DPU_DOPPLERPROC_INTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_DopplerProcHWA_DC_estimParams_t
{
    /*! @brief DC estimation scale value in Q8 format, 9bit wide (scale range 0 to 511/256) */
    uint16_t dcestScale;

    /*! @brief Programming value for DC estimation shift. Total
               right shift of the accumulator after the multiplication by dcestScale is (8+6+dcestShift) */
    uint8_t dcestShift;

    /*! @brief programming value for the scale shift in the input 
               formater, of DC estimation param set */
    uint8_t preProcScaleShift;

} DPU_DopplerProcHWA_DC_estimParams;

/**
 * @brief
 *  DopplerProc HWA DDMA Decompression internal data object
 *
 *  \ingroup DPU_DOPPLERPROC_INTERNAL_DATA_STRUCTURE
 */
typedef struct dopplerProcHWADDMADecompressionCfg
{
    /*! @brief Flag that indicates if decompression is enabled */
    bool  isEnabled;

    /*! @brief Compression Method, 0 indicates EGE */
    uint16_t  compressionMethod;
    
    /*! @brief Number of samples in a single input block to be decompressed */
    uint16_t inputSamplesPerBlock;

    /*! @brief Number of samples in a single decompressed output block */
    uint16_t outputSamplesPerBlock;

    /*! @brief Number of blocks */
    uint16_t numBlocks;

    /*! @brief Bytes per input/output sample */
    uint16_t bytesPerSample;

    /*! @brief Number of bytes per input block */
    uint16_t inputBytesPerBlock;

    /*! @brief Number of bytes per decompressed output block */
    uint16_t outputBytesPerBlock;

    /* @brief Number of Rx Antennas per block */
    uint16_t rxAntPerBlock;

    /* @brief Number of Range Bins per block */
    uint16_t rangeBinsPerBlock;

    /* @brief The actual compression ratio achieved */
    float achievedCompressionRatio;

    /* @brief Number of chirps to process per ping or pong */
    uint16_t numChirpsPerPing;

    /* @brief Number of blocks to process per ping or pong */
    uint16_t numBlocksPerPing;

    /* @brief Number of loops to run the input EDMA for */
    uint16_t numLoops;

    /* @brief Number of loops to run the input EDMA for */
    int32_t * decompEdmaToHwaStartAddress;

    uint32_t numOuterBlocks;
    uint32_t decompSizePerPingPong;
    uint32_t outerBlockSizeCompressed;

    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPongIn[2];

    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPongOut[2];

    /*! @brief  HWA Common Config */
    HWA_CommonConfig    hwaCommonConfig;

}dopplerProcHWADDMADecompressionCfg;

typedef struct dopplerProcHWADDMAIODataCfg_t
{
    /*! @brief  */
    uint8_t isReal;

    /*! @brief  */
    uint8_t bytesPerSample;

    /*! @brief  */
    uint8_t isSigned;

}dopplerProcHWADDMAIODataCfg;

typedef struct dopplerProcHWADDMADataCfg_t
{
    /*! @brief  */
    dopplerProcHWADDMAIODataCfg    input;

    /*! @brief  */
    dopplerProcHWADDMAIODataCfg    output;

}dopplerProcHWADDMADataCfg;

/**
 * @brief
 *  DopplerProc HWA DDMA Doppler/Demodulation stage internal data object
 *
 *  \ingroup DPU_DOPPLERPROC_INTERNAL_DATA_STRUCTURE
 */
typedef struct dopplerProcHWADDMADopplerDemodCfg_t
{

    /*! @brief  */
    uint16_t    numBandsTotal;

    /*! @brief  */
    uint16_t    numBandsEmpty;

    /*! @brief  */
    uint16_t    numBandsActive;

    dopplerProcHWADDMADataCfg   dopplerIOCfg;
    dopplerProcHWADDMADataCfg   logAbsIOCfg;
    dopplerProcHWADDMADataCfg   sumRxIOCfg;
    dopplerProcHWADDMADataCfg   DDMAMetricIOCfg;
    dopplerProcHWADDMADataCfg   sumTxIOCfg;

    /* @brief Number of loops to run the input EDMA for */
    uint16_t numLoops;

    /* @brief Number of loops to run the input EDMA for */
    int32_t * decompEdmaToHwaStartAddress;

    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPongIn[2];

    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPongOut[2];

    /*! @brief  HWA Common Config */
    HWA_CommonConfig    hwaCommonConfig;

}dopplerProcHWADDMADopplerDemodCfg;

/**
 * @brief
 *  CFAR Configuration
 *
 * @details
 *  The structure contains the cfar configuration used in data path
 */
typedef struct dopplerProcHWADDMAcfarCfg_t
{
    /*! @brief    CFAR threshold scale */
    uint16_t       thresholdScale;

    /*! @brief    CFAR averagining mode 0-CFAR_CA, 1-CFAR_CAGO, 2-CFAR_CASO, 3-CFAR_OS(HWA2.0 only) */
    uint8_t        averageMode;

    /*! @brief    CFAR noise averaging one sided window length */
    uint8_t        winLen;

    /*! @brief    CFAR one sided guard length*/
    uint8_t        guardLen;

    /*! @brief    CFAR cumulative noise sum divisor
                  CFAR_CA:
                        noiseDivShift should account for both left and right noise window
                        ex: noiseDivShift = ceil(log2(2 * winLen))
                  CFAR_CAGO/_CASO:
                        noiseDivShift should account for only one sided noise window
                        ex: noiseDivShift = ceil(log2(winLen))
     */
    uint8_t        noiseDivShift;

    /*! @brief    CFAR 0-cyclic mode disabled, 1-cyclic mode enabled */
    uint8_t        cyclicMode;

    /*! @brief    Peak grouping scheme 1-based on neighboring peaks from detection matrix
     *                                 2-based on on neighboring CFAR detected peaks.
     *            Scheme 2 is not supported on the HWA version (cfarprochwa.h) */
    uint8_t        peakGroupingScheme;

    /*! @brief     Peak grouping, 0- disabled, 1-enabled */
    uint8_t        peakGroupingEn;

    /*! @brief     The ordered statistic K in CFAR_OS */
    uint8_t        osKvalue;

    /*! @brief     Only used in CFAR_OS non-cyclic mode, scaling of K value for edge samples, 
     *             0- disabled, 1-enabled */
    uint8_t        osEdgeKscaleEn;

} dopplerProcHWADDMAcfarCfg;


typedef struct dopplerProcHWADDMAlocalMaxCfg_t
{
    /*! @brief    Azim threshold scale */
    uint16_t azimThreshold;

    /*! @brief    Azim threshold scale */
    uint16_t dopplerThreshold;

} dopplerProcHWADDMAlocalMaxCfg;

/**
 * @brief
 *  DopplerProc HWA DDMA Doppler/Demodulation stage internal data object
 *
 *  \ingroup DPU_DOPPLERPROC_INTERNAL_DATA_STRUCTURE
 */
typedef struct dopplerProcHWADDMAcfarAzimFFTCfg_t
{

    dopplerProcHWADDMADataCfg   cfarIOCfg;
    dopplerProcHWADDMAcfarCfg   cfarCfg;   
       
    dopplerProcHWADDMADataCfg   localMaxIOCfg;
    dopplerProcHWADDMAlocalMaxCfg   localMaxCfg;

    dopplerProcHWADDMADataCfg   azimFFTIOCfg;
    uint32_t    numAzimFFTBins;

    /* Im/Re format */
    int32_t antennaCalibParamsQuantized[DPU_DOPPLERPROCHWADDMA_MAX_NUM_TXANT * DPU_DOPPLERPROCHWADDMA_MAX_NUM_RXANT * 2];

    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPongIn[2];

    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPongOut[2];

    /*! @brief  HWA Common Config */
    HWA_CommonConfig    hwaCommonConfig;

}dopplerProcHWADDMAcfarAzimFFTCfg;

/**
 * @brief
 *  dopplerProc DPU internal data Object
 *
 * @details
 *  The structure is used to hold dopplerProc internal data object
 *
 *  \ingroup DPU_DOPPLERPROC_INTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_DopplerProcHWA_Obj_t
{
    /*! @brief HWA Handle */
    HWA_Handle  hwaHandle;
    
    /*! @brief  EDMA driver handle. */
    EDMA_Handle edmaHandle;

    /*! @brief  EDMA configuration for Input data (Radar cube -> HWA memory). */
    DPU_DopplerProc_Edma edmaIn;

    /*! @brief Decompression HWA Processing Done semaphore object */
    SemaphoreP_Object  decompHwaDoneSemaHandle;

    /*! @brief Doppler Decompression EDMA Out Done semaphore object */
    SemaphoreP_Object  decompEdmaOutDoneSemaHandle;

    /*! @brief Doppler FFT EDMA Out (ping) Done semaphore object */
    SemaphoreP_Object  dopplerFFTPingEdmaOutDoneSemaHandle;

    /*! @brief Doppler FFT EDMA Out (pong) Done semaphore object */
    SemaphoreP_Object  dopplerFFTPongEdmaOutDoneSemaHandle;

    /*! @brief DDMA Metric EDMA Out (ping) Done semaphore object */
    SemaphoreP_Object  DDMAMetricPingEdmaOutDoneSemaHandle;

    /*! @brief DDMA Metric EDMA Out (pong) Done semaphore object */
    SemaphoreP_Object  DDMAMetricPongEdmaOutDoneSemaHandle;

    /*! @brief SumTx EDMA Out (ping) Done semaphore object */
    SemaphoreP_Object  sumLogAbsPingEdmaOutDoneSemaHandle;

    /*! @brief SumTx EDMA Out (pong) Done semaphore object */
    SemaphoreP_Object  sumLogAbsPongEdmaOutDoneSemaHandle;

    /*! @brief Azim FFT EDMA Out (ping) Done semaphore object */
    SemaphoreP_Object  azimFFTPingEdmaOutDoneSemaHandle;

    /*! @brief Azim FFT EDMA Out (pong) Done semaphore object */
    SemaphoreP_Object  azimFFTPongEdmaOutDoneSemaHandle;

    /*! @brief CFAR EDMA Out (ping) Done semaphore object */
    SemaphoreP_Object  cfarPingEdmaOutDoneSemaHandle;

    /*! @brief CFAR EDMA Out (pong) Done semaphore object */
    SemaphoreP_Object  cfarPongEdmaOutDoneSemaHandle;

    /*! @brief LM EDMA Out (ping) Done semaphore object */
    SemaphoreP_Object  localMaxPingEdmaOutDoneSemaHandle;

    /*! @brief LM EDMA Out (pong) Done semaphore object */
    SemaphoreP_Object  localMaxPongEdmaOutDoneSemaHandle;
    
    /*! @brief Flag to indicate if DPU is in processing state */
    bool inProgress;
          
    /*! @brief  DMA trigger source channel for Ping/Pong param set */
    uint8_t hwaDmaTriggerSourcePingPong[2];
    
    /*! @brief  HWA number of loops */
    uint16_t hwaNumLoops;
    
    /*! @brief  HWA start paramset index */
    uint8_t  hwaParamStartIdx;
    
    /*! @brief  HWA stop paramset index */
    uint8_t  hwaParamStopIdx;
    
    /*! @brief  HWA memory bank addresses */
    uint32_t hwaMemBankAddr[DPU_DOPPLERPROCHWA_NUM_HWA_MEMBANKS];

    /*! @brief  Number of Doppler chirps. */
    uint16_t    numDopplerChirps;

    /*! @brief  Number of Doppler bins */
    uint16_t    numDopplerBins;

    /*! @brief  HWA translated memory addresses, first index for ping/pong, second index for source/destination */
    uint32_t    hwaLocMemAddr[2][2];

    /*! @brief  Decompression config */
    dopplerProcHWADDMADecompressionCfg decompCfg;

    /*! @brief  DopplerDemod config */
    dopplerProcHWADDMADopplerDemodCfg dopplerDemodCfg;

    /*! @brief CFAR-AzimFFT config */
    dopplerProcHWADDMAcfarAzimFFTCfg cfarAzimFFTCfg;

    /*! @brief DC Estimation parameters */
    DPU_DopplerProcHWA_DC_estimParams dcEstPar;

    /*! @brief Number of ping peaks detected by CFAR */
    uint32_t    numCfarPeaksPing;

    /*! @brief Number of pong peaks detected by CFAR */
    uint32_t    numCfarPeaksPong;

    /*! @brief Number of objects out */
    uint32_t    numObjOut;
    
}DPU_DopplerProcHWA_Obj;


#ifdef __cplusplus
}
#endif

#endif
