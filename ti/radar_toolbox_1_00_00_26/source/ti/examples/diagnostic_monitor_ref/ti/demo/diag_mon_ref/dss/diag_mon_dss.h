/**
 *   @file  diag_mon_dss.h
 *
 *   @brief
 *      This is the main header file for Diagnostic & Monitor Demo
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
#ifndef DIAG_MON_DSS_H
#define DIAG_MON_DSS_H

#include <ti/drivers/soc/soc.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/mailbox/mailbox.h>
#include "osal/DebugP.h"
#include "osal/src/nonos/osal_Event_nonos.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief
 *  Millimeter Wave Demo state
 *
 * @details
 *  The enumeration is used to hold the data path states for the
 *  Millimeter Wave demo
 */
typedef enum MmwDemo_DSS_STATE_e
{
    /*! @brief   State after data path is initialized */
    MmwDemo_DSS_STATE_INIT = 0,

    /*! @brief   State after data path is started */
    MmwDemo_DSS_STATE_STARTED,

    /*! @brief   State after data path is stopped */
    MmwDemo_DSS_STATE_STOPPED,

    /*! @brief   State after STOP request was received by DSP
                 but complete stop is on-going */
    MmwDemo_DSS_STATE_STOP_PENDING

}MmwDemo_DSS_STATE;

/**
 * @brief
 *  Millimeter Wave Demo MCB
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Millimeter Wave demo
 */
typedef struct MmwDemo_DSS_MCB_t
{
    /*! * @brief   Handle to the SOC Module */
    SOC_Handle                  socHandle;

    /*!@brief   Handle to the peer Mailbox */
    Mbox_Handle              peerMailbox;

    /*! @brief   Semaphore handle for the mailbox communication */
    SemaphoreP_Handle            mboxSemHandle;

    /*! @brief   DSS event handle */
    Event_Handle                eventHandle;

    /*! @brief   mmw Demo state */
    MmwDemo_DSS_STATE           state;

    /*! @brief   Diagnostic Test status message */
    MmwDemo_message         diagStatusMsg;
} MmwDemo_DSS_MCB;


/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/
/* Diagnostic test Functions */
extern int32_t DssDiag_InjectTest(void);

/* Sensor Management Module Exported API */
extern void _MmwDemo_debugAssert(int32_t expression, const char *file, int32_t line);
#define MmwDemo_debugAssert(expression) {                                      \
                                         DebugP_assert(expression);             \
                                        }
                                        
#ifdef __cplusplus
}
#endif

#endif /* DIAG_MON_DSS_H */

