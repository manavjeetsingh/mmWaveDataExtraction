/**
 *   @file  mmw_output.h
 *
 *   @brief
 *      This is the interface/message header file for the Millimeter Wave Demo
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2016 Texas Instruments, Inc.
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
#ifndef MMW_OUTPUT_H
#define MMW_OUTPUT_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Output packet length is a multiple of this value, must be power of 2*/
#define MMW_OUTPUT_MSG_SEGMENT_LEN  16

#define MESSAGE_MAGIC_WORD1      0x1234
#define MESSAGE_MAGIC_WORD2      0x5678


/** 16 Bit Type: [15:8] async_event_msg, [7:0] monitor type */
#define MMW_RF_INIT_CALIB_STATUS        (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_INITCALIBSTATUS_SB)
#define MMW_RF_RUNTIME_CALIB_REP        (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_RUN_TIME_CALIB_REPORT_SB)
#define MMW_RF_DIG_LATENTFAULT_REP      (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_DIG_LATENTFAULT_REPORT_SB)
#define MMW_RF_TEMPERATURE_REP          (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_TEMPERATURE_REPORT_SB)
#define MMW_RF_RX_GAIN_PHASE_REP        (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_RX_GAIN_PHASE_REPORT)
#define MMW_RF_RX_NOISE_FIG_REP         (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_RX_NOISE_FIG_REPORT)
#define MMW_RF_RX_IF_STAGE_REP          (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_RX_IF_STAGE_REPORT)
#define MMW_RF_TX0_POWER_REP            (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_TX0_POWER_REPORT)
#define MMW_RF_TX1_POWER_REP            (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_TX1_POWER_REPORT)
#define MMW_RF_TX2_POWER_REP            (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_TX2_POWER_REPORT)
#define MMW_RF_TX0_BALLBRK_REP          (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_TX0_BALLBREAK_REPORT)
#define MMW_RF_TX1_BALLBRK_REP          (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_TX1_BALLBREAK_REPORT) 
#define MMW_RF_TX2_BALLBRK_REP          (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_TX2_BALLBREAK_REPORT)
#define MMW_RF_DIG_PERIODIC_REP         (uint16_t)((RL_RF_ASYNC_EVENT_MSG << 8) | RL_RF_AE_MON_DIG_PERIODIC_REPORT_SB) 
 
#define MMW_RF_TX_GAIN_MISMATCH_REP     (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_TX_GAIN_MISMATCH_REPORT)
#define MMW_RF_SYNTHESIZER_FREQ_REP     (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_SYNTHESIZER_FREQ_REPORT)
#define MMW_RF_EXT_ANALOG_SIG_REP       (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_EXT_ANALOG_SIG_REPORT)
#define MMW_RF_PMCLKLO_INT_ANA_SIG_REP  (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_PMCLKLO_INT_ANA_SIG_REPORT)
#define MMW_RF_GPADC_INT_ANA_SIG_REP    (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_GPADC_INT_ANA_SIG_REPORT)
#define MMW_RF_PLL_CONTROL_VOLT_REP     (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_PLL_CONTROL_VOLT_REPORT)
#define MMW_RF_DCC_CLK_FREQ_REP         (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_DCC_CLK_FREQ_REPORT)
#define MMW_RF_TX0_INT_ANA_SIG_REP      (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_TX0_INT_ANA_SIG_REPORT)
#define MMW_RF_TX1_INT_ANA_SIG_REP      (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_TX1_INT_ANA_SIG_REPORT)
#define MMW_RF_TX2_INT_ANA_SIG_REP      (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_TX2_INT_ANA_SIG_REPORT)
#define MMW_RF_RX_INT_ANA_SIG_REP       (uint16_t)((RL_RF_ASYNC_EVENT_1_MSG << 8) | RL_RF_AE_MON_RX_INT_ANALOG_SIG_REPORT)

/*!
 * @brief
 *  Message header for reporting detection information from data path.
 *
 * @details
 *  The structure defines the message header.
 */
typedef struct MmwDemo_output_message_header_t
{
    /*! @brief   Output buffer magic word (sync word). It is initialized to  {0x1234,0x5678} */
    uint16_t    magicWord[2];

    /*! brief   Version: : MajorNum * 2^24 + MinorNum * 2^16 + BugfixNum * 2^8 + BuildNum   */
    uint32_t     version;

    /*! @brief   Total packet length including header in Bytes */
    uint32_t    totalPacketLen;
    
    /*! @brief   Frame number */
    uint32_t    frameNumber;

    /*! @brief   platform type */
    uint32_t     platform;
    
    /*! @brief   Number of TLVs */
    uint32_t    numTLVs;
} MmwDemo_output_message_header;


/**
 * @brief
 *  Message for reporting detected objects from data path.
 *
 * @details
 *  The structure defines the message body for detected objects from from data path. 
 */
typedef struct MmwDemo_output_message_tl_t
{
    /*! @brief   TLV type */
    uint16_t    type;
    
    /*! @brief   Length in bytes */
    uint16_t    length;

} MmwDemo_output_message_tl;

/*!
 * @brief
 *  Message types used in Millimeter Wave Demo for the communication between
 *  target and host, and also for Mailbox communication
 *  between MSS and DSS on the XWR18xx platform. Message types are used to indicate
 *  different type detection information sent out from the target.
 *
 */
typedef enum MmwDemo_output_message_type_enum
{
    /*! @brief   List of detected points */
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1,

    /*! @brief   Stats information */
    MMWDEMO_OUTPUT_MSG_STATS,
    

    MMWDEMO_OUTPUT_MSG_MAX = 21 /* max no. of report */
} MmwDemo_output_message_type_;

/*!
 * @brief
 * Structure holds message stats information from data path.
 *
 * @details
 *  The structure holds stats information. This is a payload of the TLV message item
 *  that holds stats information.
 */
typedef struct MmwDemo_output_message_stats_t
{
    /*! @brief   Interframe processing time in usec */
    uint32_t     interFrameProcessingTime;

    /*! @brief   Transmission time of output detection information in usec */
    uint32_t     transmitOutputTime;

    /*! @brief   Interframe processing margin in usec */
    uint32_t     interFrameProcessingMargin;

    /*! @brief   Interchirp processing margin in usec */
    uint32_t     interChirpProcessingMargin;
    
    /*! @brief   CPU Load (%) during active frame duration */
    uint32_t     activeFrameCPULoad;
    
    /*! @brief   CPU Load (%) during inter frame duration */
    uint32_t     interFrameCPULoad;
} MmwDemo_output_message_stats;

/**
 * @brief
 *  Size of HSRAM Payload data array.
 */
#define MMWDEMO_HSRAM_PAYLOAD_SIZE        (SOC_HSRAM_SIZE - sizeof(DPC_ObjectDetection_ExecuteResult) - \
                                            sizeof(MmwDemo_output_message_stats) - MAX_MEMSIZE_SYSVAL)

#ifdef __cplusplus
}
#endif

#endif /* MMW_OUTPUT_H */
