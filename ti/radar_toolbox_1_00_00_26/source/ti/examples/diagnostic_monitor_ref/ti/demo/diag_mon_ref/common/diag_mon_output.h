/**
 *   @file  diag_mon_output.h
 *
 *   @brief
 *      This is the interface/message header file for the Diagnostic & 
 *      Monitoring Demo
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
#ifndef DIAG_MON_OUTPUT_H
#define DIAG_MON_OUTPUT_H

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

/* max no. of report */
#define MMWDEMO_OUTPUT_MSG_MAX   21

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

#define MMWDEMO_MAX_FILE_NAME_SIZE 128
/**
 * @brief
 *  Message for reporting DSS assertion information
 *
 * @details
 *  The structure defines the message body for the information
 *  on a DSS exception that should be forwarded to the MSS.
 */
typedef struct MmwDemo_dssAssertInfoMsg_t
{
    /*! @brief file name */
    char     file[MMWDEMO_MAX_FILE_NAME_SIZE];

    /*! @brief line number */
    uint32_t line;
} MmwDemo_dssAssertInfoMsg;

typedef struct MmwDemo_dssStatusMsg_t
{
    uint16_t   eventType;
    int32_t    errVal;
}MmwDemo_dssStatusMsg;

typedef struct MmwDemo_dssDiagTestMsg_t
{
    uint32_t  diagTestBitStat;
    uint32_t  diagTestExecBits;
    int32_t   errVal;
}MmwDemo_dssDiagTestMsg;

/**
 * @brief
 *  Message body used in Millimeter Wave Demo for passing configuration from MSS
 * to DSS.
 *
 * @details
 *  The union defines the message body for various configuration messages.
 */
typedef union MmwDemo_message_body_u
{
    /*! @brief   DSS assertion information */
    MmwDemo_dssAssertInfoMsg  assertInfo;

    /*! @brief   Misc DSS status info */
    MmwDemo_dssStatusMsg      dssStatusInfo;

    /*! @brief   DSS Diagnostic test status info */
    MmwDemo_dssDiagTestMsg    dssDiagTestStat;
} MmwDemo_message_body;

/**
 * @brief
 *  Message types used in Millimeter Wave Demo for Mailbox communication 
 * between MSS and DSS.
 *
 * @details
 *  The enum is used to hold all the messages types used for Mailbox communication
 * between MSS and DSS in mmw Demo.
 */
typedef enum MmwDemo_message_type_e 
{
    /*! @brief   message types for MSS to DSS communication */
    MMWDEMO_MSS2DSS_TEST_STAT_REQ = 0xFEED0001,
    MMWDEMO_MSS2DSS_TEST_EXEC_REQ,

    /*! @brief   message types for DSS to MSS communication */
    MMWDEMO_DSS2MSS_DIAG_STATUS = 0xFEED0100,
    MMWDEMO_DSS2MSS_STATUS_INFO,
    MMWDEMO_DSS2MSS_ASSERT_INFO
}MmwDemo_message_type;

/**
 * @brief
 *  DSS/MSS communication messages
 *
 * @details
 *  The structure defines the message structure used to commuincate between MSS
 * and DSS.
 */
typedef struct MmwDemo_message_t
{
    /*! @brief   message type */
    MmwDemo_message_type      type;

    /*! @brief  message body */
    MmwDemo_message_body      body;

} MmwDemo_message;


/**
 * @brief
 *  Size of HSRAM Payload data array.
 */
#define MMWDEMO_HSRAM_PAYLOAD_SIZE        (SOC_HSRAM_SIZE - sizeof(DPC_ObjectDetection_ExecuteResult) - \
                                            sizeof(MmwDemo_output_message_stats) - MAX_MEMSIZE_SYSVAL)

#ifdef __cplusplus
}
#endif

#endif /* DIAG_MON_OUTPUT_H */
