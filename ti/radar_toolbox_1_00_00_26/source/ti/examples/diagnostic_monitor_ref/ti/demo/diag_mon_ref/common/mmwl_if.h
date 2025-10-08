/*
 *   @file  mmwl_if.h
 *
 *   @brief
 *      Header file for mmwl_if.c which test the mmWave Link API
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

#ifndef MMWL_IF_H
#define MMWL_IF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Bit manipulations */
#define GET_BIT_VALUE(data, noOfBits, location)    ((((rlUInt32_t)(data)) >> (location)) &\
                                               (((rlUInt32_t)((rlUInt32_t)1U << (noOfBits))) - (rlUInt32_t)1U))

/*Maximum number of different monitoring reports*/
#define MMWAVELINK_TEST_MAX_NUM_MON 40

#define MMWAVELINK_TEST_MON_TEMP       0U
#define MMWAVELINK_TEST_MON_RX_GAIN_PH 1U


/**
 * @brief
 *  Mmwave Link Master Control Block
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Mmwave Link.
 */
typedef struct MmwaveLink_MCB
{
    /**
     * @brief   Handle to the BSS Mailbox
     */
    Mbox_Handle              bssMailbox;

    /**
     * @brief   Semaphore handle for the mmWave Link
     */
    Semaphore_Handle            linkSemaphore;

    /**
     * @brief   mmWave Link Spawning function
     */
    RL_P_OSI_SPAWN_ENTRY        spawnFxn;

    /**
     * @brief   Status of the BSS:
     */
    volatile uint32_t           bssStatus;

    /**
     * @brief   Counter which tracks of the number of times the spawn function was
     * overrun.
     */
    uint32_t                    spawnOverrun;
    /**
     * @brief   Handle to the CRC Channel
     */
    CRC_Handle                  crcHandle;
}MmwaveLink_MCB;

/**
 * @brief
 *  Millimeter Wave Demo statistics
 *
 * @details
 *  The structure is used to hold the statistics information for the
 *  Millimeter Wave demo
 */
typedef struct MMWL_Stats_t
{
    /*! @brief   Counter which tracks the number of frame trigger events from BSS */
    uint64_t     frameTriggerReady;

    /*! @brief   Counter which tracks the number of failed calibration reports
     *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     failedTimingReports;

    /*! @brief   Counter which tracks the number of calibration reports received
     *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     calibrationReports;

     /*! @brief   Counter which tracks the number of sensor stop events received
      *           The event is triggered by an asynchronous event from the BSS */
    uint32_t     sensorStopped;
}MMWL_Stats;


extern int32_t MmwaveLink_initLink (rlUInt8_t deviceType, rlUInt8_t platform);
extern int32_t MmwaveLink_getRfBootupStatus (rlRfBootStatusCfg_t *ptrStatusCfg);
extern int32_t MmwaveLink_getVersion (void);
extern int32_t MmwaveLink_frameTrigger(uint8_t waitAe);
extern int32_t MmwaveLink_Config(void);
extern int32_t MmwaveLink_frameStop(uint8_t waitAe);
extern void MmwaveLink_initCfg(void);

#ifdef __cplusplus
}
#endif

#endif
