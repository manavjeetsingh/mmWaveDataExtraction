/*
 *  
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
 
 /**
 *   @file  rangeprochwaDDMA.h
 *
 *   @brief
 *      Implements range processing functionality using HWA.
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/
#ifndef RANGEPROCHWA_H
#define RANGEPROCHWA_H

/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

/* MCU Plus SDK Drivers include files */
#include <drivers/hwa.h>

/* mmWave SDK Data Path Include Files */
#include <ti/datapath/dpif/dpif_adcdata.h>
#include <ti/datapath/dpif/dpif_radarcube.h>
#include <ti/datapath/dpif/dp_error.h>
#include <ti/datapath/dpedma/dpedmahwa.h>
#include <ti/datapath/dpedma/dpedma.h>
#include "../rangeproc/rangeproc_common.h"

#ifdef __cplusplus
extern "C" {
#endif
 
/*! Number of HWA parameter sets */
#define DPU_RANGEPROCHWA_NUM_HWA_PARAM_SETS             4U
#define DPU_RANGEPROCHWA_NUM_HWA_PARAM_SETS_DDMA        10U //TODO Add options

/*! Alignment for DC range signal mean buffer - if DPU is running on DSP(C66) */
#define DPU_RANGEPROCHWA_DCRANGESIGMEAN_BYTE_ALIGNMENT_DSP 8U

/*! Alignment for DC range signal mean buffer - if DPU is running on R5F */
#define DPU_RANGEPROCHWA_DCRANGESIGMEAN_BYTE_ALIGNMENT_R5F 4U

/*! Alignment for radar cube on R5F */
#define DPU_RANGEPROCHWA_RADARCUBE_BYTE_ALIGNMENT_R5F    CSL_CACHE_L1D_LINESIZE

/*! Alignment for radar cube on DSP */
#define DPU_RANGEPROCHWA_RADARCUBE_BYTE_ALIGNMENT_DSP    (sizeof(int16_t))

#define DPU_RANGEPROCHWADDMA_NUM_INTFMITIG_WIN_HWACOMMONCFG_SIZE (5U)

#define RANGEPROCHWADDMA_NUM_EDMA_INTERRUPTS 1U

/** @addtogroup DPU_RANGEPROC_ERROR_CODE
 *  Base error code for the rangeProc DPU is defined in the
 *  \include ti/datapath/dpif/dp_error.h
 @{ */

/**
 * @brief   Error Code: Invalid argument
 */
#define DPU_RANGEPROCHWA_EINVAL                  (DP_ERRNO_RANGE_PROC_BASE-1)

/**
 * @brief   Error Code: Out of memory
 */
#define DPU_RANGEPROCHWA_ENOMEM                  (DP_ERRNO_RANGE_PROC_BASE-2)

/**
 * @brief   Error Code: Internal error
 */
#define DPU_RANGEPROCHWA_EINTERNAL               (DP_ERRNO_RANGE_PROC_BASE-3)

/**
 * @brief   Error Code: Not implemented
 */
#define DPU_RANGEPROCHWA_ENOTIMPL                (DP_ERRNO_RANGE_PROC_BASE-4)

/**
 * @brief   Error Code: In Progress
 */
#define DPU_RANGEPROCHWA_EINPROGRESS             (DP_ERRNO_RANGE_PROC_BASE-5)

/**
 * @brief   Error Code: Invalid control command
 */
#define DPU_RANGEPROCHWA_ECMD                    (DP_ERRNO_RANGE_PROC_BASE-6)

/**
 * @brief   Error Code: Semaphore error
 */
#define DPU_RANGEPROCHWA_ESEMA                   (DP_ERRNO_RANGE_PROC_BASE-7)

/**
 * @brief   Error Code: DC range signal removal configuration error
 */
#define DPU_RANGEPROCHWA_EDCREMOVAL              (DP_ERRNO_RANGE_PROC_BASE-8)

/**
 * @brief   Error Code: ADCBuf data interface configuration error
 */
#define DPU_RANGEPROCHWA_EADCBUF_INTF            (DP_ERRNO_RANGE_PROC_BASE-9)

/**
 * @brief   Error Code: ADCBuf data interface configuration error
 */
#define DPU_RANGEPROCHWA_ERADARCUBE_INTF         (DP_ERRNO_RANGE_PROC_BASE-10)

/**
 * @brief   Error Code: HWA windowing configuration error
 */
#define DPU_RANGEPROCHWA_EWINDOW                 (DP_ERRNO_RANGE_PROC_BASE-11)

/**
 * @brief   Error Code: Incorrect number of butterfly stages specified for scaling 
 */
#define DPU_RANGEPROCHWA_EBUTTERFLYSCALE         (DP_ERRNO_RANGE_PROC_BASE-12)



/**
@}
*/

/**
 * @brief
 *  RangeProc data input mode
 *
 * @details
 *  This enum defines if the rangeProc input data is from RF front end or it is in M0 but 
 *  standalone from RF.
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
// typedef enum DPU_RangeProcHWA_InputMode_e
// {
//     /*! @brief     Range input is integrated with DFE input 
//                     ADC buffer is mapped to HWA memory 
//                     DMA data from ADC buffer to HWA is NOT required
//                     This is not supported in TPR12 */
//     DPU_RangeProcHWA_InputMode_MAPPED,

//     /*! @brief     Range input is integrated with DFE input 
//                     ADC buffer is not mapped to HWA memory,
//                     DMA data from ADCBuf to HWA memory is 
//                     needed in range processing */
//     DPU_RangeProcHWA_InputMode_ISOLATED,

//     /*! @brief      Range input is stored into HWA internal memory 
//                     in ping/pong manner*/
//     DPU_RangeProcHWA_InputMode_HWA_INTERNAL_MEM
// } DPU_RangeProcHWA_InputMode;

typedef uint32_t DPU_RangeProcHWA_InputMode;
#define DPU_RangeProcHWA_InputMode_MAPPED              (uint32_t) 0U
#define DPU_RangeProcHWA_InputMode_ISOLATED            (uint32_t) 1U  
#define DPU_RangeProcHWA_InputMode_HWA_INTERNAL_MEM    (uint32_t) 2U

/**
 * @brief
 *  rangeProc control command
 *
 * @details
 *  The enum defines the rangeProc supported run time command
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
// typedef enum DPU_RangeProcHWA_Cmd_e
// {
//     /*! @brief     Command to update DC Signature removal configuration */
//     DPU_RangeProcHWA_Cmd_dcRangeCfg,

//     /*! @brief     Command to trigger rangeProcHWA process */
//     DPU_RangeProcHWA_Cmd_triggerProc,
// }DPU_RangeProcHWA_Cmd;
typedef uint32_t DPU_RangeProcHWA_Cmd;
#define DPU_RangeProcHWA_Cmd_dcRangeCfg         (uint32_t) 0
#define DPU_RangeProcHWA_Cmd_triggerProc        (uint32_t) 1

/**
 * @brief
 *  rangeProc FFT tuning parameters for HWA based Range FFT
 *
 * @details
 *  This structure allows users to tune the scaling factors for HWA based Range FFTs
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_FFTtuning_t{
    /*! @brief  Specify amount of right (divide) shift to apply
           to convert HWA internal 24-bit Range FFT output to 16-bit RadarCube.
           User should adjust this based on the setup where sensor is deployed and
           sensors setting for Tx O/P power/RX gain and their application needs */
    uint16_t    fftOutputDivShift;

    /*! @brief  Specify number of Last butterfly stages to scale to avoid clipping within 
           HWA FFT stages. Given the ADC data bit width of 16-bits and internal 24-bit width
           of HWA, user has around 8-bits to grow Range FFT output and should not need to use butterfly scaling
           for FFT sizes upto 256. Beyond that fft size, user should adjust this based on the setup 
           where sensor is deployed and sensors setting for Tx O/P power/RX gain*/
    uint16_t    numLastButterflyStagesToScale;

}DPU_RangeProcHWA_FFTtuning;

/**
 * @brief
 *  RangeProc HWA configuration
 *
 * @details
 *  The structure is used to hold the HWA configuration needed for Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_HwaConfig_t
{
    /*! @brief     HWA paramset Start index */
    uint8_t         paramSetStartIdx;

    /*! @brief     Number of HWA param sets must be @ref DPU_RANGEPROCHWA_NUM_HWA_PARAM_SETS */
    uint8_t         numParamSet;

    /*! @brief     Flag to indicate if HWA windowing is symmetric
                    see HWA_WINDOW_SYMM definitions in HWA driver's doxygen documentation
     */
    uint8_t         hwaWinSym;    

    /*! @brief     HWA windowing RAM offset in number of samples */
    uint16_t        hwaWinRamOffset;

    /*! @brief     Data Input Mode, */
    DPU_RangeProcHWA_InputMode      dataInputMode;

    /*! @brief Pointer to HWA Interference Mitigation window */
    uint8_t         hwaInterfMitigWindow[DPU_RANGEPROCHWADDMA_NUM_INTFMITIG_WIN_HWACOMMONCFG_SIZE];

    /*! @brief     HWA hardware trigger source. This is used only in @ref DPU_RangeProcHWA_InputMode_HWA_INTERNAL_MEM mode */
    uint8_t         hardwareTrigSrc;

}DPU_RangeProcHWA_HwaConfig;

/**
 * @brief
 *  RangeProc EDMA configuration
 *
 * @details
 *  The structure is used to hold the EDMA configuration needed for Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_EDMAInputConfig_t
{
    /*! @brief     EDMA configuration for rangeProc data Input
                    This is needed only in @ref DPU_RangeProcHWA_InputMode_ISOLATED
     */
    DPEDMA_ChanCfg        dataIn;

    /*! @brief     EDMA configuration for rangeProc data Input Signature */
    DPEDMA_ChanCfg        dataInSignature;
}DPU_RangeProcHWA_EDMAInputConfig;

/**
 * @brief
 *  RangeProc EDMA configuration
 *
 * @details
 *  The structure is used to hold the EDMA configuration needed for Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_EDMAOutputConfigFmt1_t
{
    /*! @brief     EDMA configuration for rangeProc data Out- ping 
                    It must be a HWACC triggered EDMA channel.
     */
    DPEDMA_ChanCfg        dataOutPing;

    /*! @brief     EDMA configuration for rangeProc data Out- pong 
                    It must be a HWACC triggered EDMA channel
     */
    DPEDMA_ChanCfg        dataOutPong;
}DPU_RangeProcHWA_EDMAOutputConfigFmt1;

/**
 * @brief
 *  RangeProc EDMA configuration
 *
 * @details
 *  The structure is used to hold the EDMA configuration needed for Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_EDMAOutputConfigFmt2_t
{
    /*! @brief     EDMA configuration for rangeProc data Out- ping 
                    It must be a HWACC triggered EDMA channel
     */
    DPEDMA_3LinkChanCfg   dataOutPing;
    DPEDMA_ChanCfg        dataOutPingData[3];

    /*! @brief     EDMA configuration for rangeProc data Out- pong 
                    It must be a HWACC triggered EDMA channel
     */
    DPEDMA_3LinkChanCfg   dataOutPong;
    DPEDMA_ChanCfg        dataOutPongData[3];
}DPU_RangeProcHWA_EDMAOutputConfigFmt2;

/**
 * @brief
 *  RangeProc output EDMA configuration
 *
 * @details
 *  The structure is used to hold the EDMA configuration needed for Range FFT
 *
 *  Fmt1: Generic EDMA ping/pong output mode
 *       - 1 ping/pong EDMA channel, 
 *       - 1 ping/pong HWA signature channel
 *
 *  Fmt2: Specific EDMA ping/pong output mode used ONLY for 3 TX anntenna for radar cube
 *        layout format: @ref DPIF_RADARCUBE_FORMAT_1, ADCbuf interleave mode 
 *        @ref DPIF_RXCHAN_NON_INTERLEAVE_MODE
 *       - 1 ping/pong dummy EDMA channel with 3 shadow channels
         - 3 ping/pong dataOut channel
 *       - 1 ping/pong HWA signature channel 
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_EDMAOutputConfig_t
{
    /*! @brief     EDMA data output Signature */
    DPEDMA_ChanCfg        dataOutSignature;

    union
    {
        /*! @brief     EDMA data output fmt1 @ref DPU_RangeProcHWA_EDMAOutputConfigFmt1 */
        DPU_RangeProcHWA_EDMAOutputConfigFmt1     fmt1;

        /*! @brief     EDMA data output fmt2 @ref DPU_RangeProcHWA_EDMAOutputConfigFmt2 */
        DPU_RangeProcHWA_EDMAOutputConfigFmt2     fmt2;
    }u;
}DPU_RangeProcHWA_EDMAOutputConfig;

/**
 * @brief
 *  RangeProcHWA hardware resources
 *
 * @details
 *  The structure is used to hold the hardware resources needed for Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_HW_Resources_t
{
    /*! @brief     EDMA Handle */
    EDMA_Handle         edmaHandle;

    /*! @brief     HWA configuration */
    DPU_RangeProcHWA_HwaConfig      hwaCfg;

    /*! @brief     EDMA configuration for rangeProc data Input */
    DPU_RangeProcHWA_EDMAInputConfig edmaInCfg;

    /*! @brief     EDMA configuration for rangeProc data Output */ 
    DPU_RangeProcHWA_EDMAOutputConfig edmaOutCfg;

    /*! @brief     Pointer to Calibrate DC Range signature buffer 
                    The size of the buffer = DPU_RANGEPROC_SIGNATURE_COMP_MAX_BIN_SIZE *
                                        numTxAntenna * numRxAntenna * sizeof(cmplx32ImRe_t)
        For R5F:\n
        Byte alignment Requirement = @ref DPU_RANGEPROCHWA_DCRANGESIGMEAN_BYTE_ALIGNMENT_R5F \n
        For DSP (C66X):\n
        Byte alignment Requirement = @ref DPU_RANGEPROCHWA_DCRANGESIGMEAN_BYTE_ALIGNMENT_DSP \n
     */
    cmplx32ImRe_t       *dcRangeSigMean;

    /*! @brief     DC range calibration scratch buffer size */
    uint32_t            dcRangeSigMeanSize;

    /*! @brief      Radar cube data interface. Radar cube buffer (radarCube.data)
        For R5F:\n
        Byte alignment Requirement = @ref DPU_RANGEPROCHWA_RADARCUBE_BYTE_ALIGNMENT_R5F \n
        For DSP (C66X):\n
        Byte alignment Requirement = @ref DPU_RANGEPROCHWA_RADARCUBE_BYTE_ALIGNMENT_DSP \n
     */
    DPIF_RadarCube      radarCube;

    /* EDMA Interrupt Object */
    Edma_IntrObject     *edmaTransferCompleteIntrObj;

}DPU_RangeProcHWA_HW_Resources;

/**
 * @brief
 *  RangeProcHWA Compression hardware resources
 *
 * @details
 *  The structure is used to hold the hardware resources needed for compression of Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_CompressionCfg
{
    /*! @brief Flag that indicates if compression is enabled */
    bool  isEnabled;

    /*! @brief Compression Method, 0 indicates EGE and 1 indicates BFP */
    uint16_t  compressionMethod;

    /*! @brief Compression ration, a value between 0 and 1 */
    float  compressionRatio;

    /*! @brief Indicates the number of range bins to be compressed in a single compression block */
    uint16_t  rangeBinsPerBlock;

    /*! @brief Can be greater than 1 only for DPIF_RADARCUBE_FORMAT_2. 
               For DPIF_RADARCUBE_FORMAT_1 this should be set to 1 */
    uint16_t  numRxAntennaPerBlock;

}DPU_RangeProcHWA_CompressionCfg;

/**
 * @brief
 *  RangeProcHWA Interference mitigation configuration
 *
 * @details
 *  The structure is used to hold the hardware resources needed for interference mitigation in Range DPU
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWADDMA_intfStatsdBCfg_t
{

    /*! @brief Interference mitigation magnitude SNR in dB */
    uint32_t intfMitgMagSNRdB;

    /*! @brief Interference mitigation magdiff SNR in dB */
    uint32_t intfMitgMagDiffSNRdB;

}DPU_RangeProcHWADDMA_intfStatsdBCfg;

/**
 * @brief
 *  RangeProcHWA static configuration
 *
 * @details
 *  The structure is used to hold the static configuraiton used by rangeProcHWA
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_StaticConfig_t
{
    /*! @brief  Number of transmit antennas */
    uint8_t     numTxAntennas;

    /*! @brief  Number of virtual antennas */
    uint8_t     numVirtualAntennas;

    /*! @brief  Number of range bins */
    uint16_t    numRangeBins;

    /*! @brief  Number of bins used in Range FFT Calculation */
    uint16_t    numFFTBins;

    /*! @brief  1 if ADC Samples are real */
    uint16_t    isChirpDataReal;

    /*! @brief  Number of chirps per frame */
    uint16_t    numChirpsPerFrame;

    /*! @brief  Range FFT window coefficients, Appliation provided windows coefficients
                After @ref DPU_RangeProcHWA_config(), windowing buffer is not used by rangeProcHWA DPU,
                Hence memory can be released
     */
    int32_t    *window;

    /*! @brief     Range FFT window coefficients size in bytes 
                    non-symmetric window, size = sizeof(uint32_t) * numADCSamples
                    symmetric window, size = sizeof(uint32_t)*(numADCSamples round up to even number )/2
     */
    uint32_t    windowSize;

    /*! @brief      ADCBuf buffer interface */
    DPIF_ADCBufData     ADCBufData;

    /*! @brief      Flag to reset dcRangeSigMean buffer
                     1 - to reset the dcRangeSigMean buffer and counter
                     0 - do not reset
     */
    uint8_t     resetDcRangeSigMeanBuffer;

    /*! @brief     Range FFT Tuning Params */
    DPU_RangeProcHWA_FFTtuning    rangeFFTtuning;

    /*! @brief Compression Configuration */
    DPU_RangeProcHWA_CompressionCfg  compressionCfg;

    /*! @brief Interference mitigation Configuration */
    DPU_RangeProcHWADDMA_intfStatsdBCfg intfStatsCfgdB;

}DPU_RangeProcHWA_StaticConfig;

/**
 * @brief
 *  RangeProcHWA dynamic configuration
 *
 * @details
 *  The structure is used to hold the dynamic configuraiton used by rangeProcHWA
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_DynamicConfig_t
{
    /*! @brief      Pointer to Calibrate DC Range signature configuration */
    DPU_RangeProc_CalibDcRangeSigCfg *calibDcRangeSigCfg;
}DPU_RangeProcHWA_DynamicConfig;

/**
 * @brief
 *  Range FFT configuration
 *
 * @details
 *  The structure is used to hold the configuration needed for Range FFT
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_Config_t
{
    /*! @brief     rangeProc hardware resources */
    DPU_RangeProcHWA_HW_Resources   hwRes;

    /*! @brief     rangeProc static configuration */
    DPU_RangeProcHWA_StaticConfig   staticCfg;

    /*! @brief     rangeProc dynamic configuration */
    DPU_RangeProcHWA_DynamicConfig  dynCfg;
}DPU_RangeProcHWA_Config;

/**
 * @brief
 *  rangeProcHWA output parameters populated during rangeProc Processing time
 *
 * @details
 *  The structure is used to hold the output parameters for rangeProcHWA
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_InitParams_t
{
    /*! @brief     HWA Handle */
    HWA_Handle          hwaHandle;
}DPU_RangeProcHWA_InitParams;

/**
 * @brief
 *  rangeProcHWA output parameters populated during rangeProc Processing time
 *
 * @details
 *  The structure is used to hold the output parameters for rangeProcHWA
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef struct DPU_RangeProcHWA_OutParams_t
{
    /*! @brief      End of Chirp indication for rangeProcHWA */
    bool                endOfChirp;

    /*! @brief     rangeProcHWA stats */
    DPU_RangeProc_stats  stats;
}DPU_RangeProcHWA_OutParams;

/**
 * @brief
 *  rangeProc DPU Handle
 *
 *  \ingroup DPU_RANGEPROC_EXTERNAL_DATA_STRUCTURE
 */
typedef void* DPU_RangeProcHWA_Handle ;

/*================================================================
               rangeProcHWA DPU exposed APIs            
 ================================================================*/
DPU_RangeProcHWA_Handle DPU_RangeProcHWA_init
(
    DPU_RangeProcHWA_InitParams     *initParams,
    int32_t*                        errCode
);

int32_t DPU_RangeProcHWA_config
(
    DPU_RangeProcHWA_Handle     handle,
    DPU_RangeProcHWA_Config*    rangeHwaCfg
);

int32_t DPU_RangeProcHWA_process
(
    DPU_RangeProcHWA_Handle     handle,
    DPU_RangeProcHWA_OutParams* outParams
);

int32_t DPU_RangeProcHWA_control
(
    DPU_RangeProcHWA_Handle handle,
    DPU_RangeProcHWA_Cmd    cmd,
    void*                   arg,
    uint32_t                argSize
);

int32_t DPU_RangeProcHWA_deinit
(
    DPU_RangeProcHWA_Handle     handle
);

#ifdef __cplusplus
}
#endif

#endif
