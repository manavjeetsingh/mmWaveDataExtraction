/*
 *   @file  osal_r4f_esmdrv.c
 *
 *   @brief
 *          This file implements the R4F TI-RTOS OSAL ESM Driver Interface
 *          File is mainly used for Safety Diagnostic Library (SDL).
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

/* Standard Include Files. */
#include <string.h>

/* mmWave SDL Include Files */
#include "osal/osal.h"
#include <ti/csl/csl.h>
#include <osal/mmwave_sdk_port.h>
#include <ti/drivers/esm/esm.h>

ESM_Handle  gEsmHandle;

/**************************************************************************
 **************************** Local Declarations **************************
 **************************************************************************/

/**************************************************************************
 **************************** Extern Declarations *************************
 **************************************************************************/

/* This function is used only by the Test code to initialize the OSAL R4F
 * ESM Driver module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_R4F_ESMDrv_init (void);

/**************************************************************************
 **************************** Local Definitions ***************************
 **************************************************************************/

/**
 * @brief   Maximum number of Group1 ESM Error channels: As per the TRM
 * this is limited to 64
 */
#define ESM_MAX_GROUP1_ERROR_CHANNEL            ((uint8_t)64U)

/**
 * @brief   Maximum number of Group2 ESM Error channels: As per the TRM
 * this is limited to 32
 */
#define ESM_MAX_GROUP2_ERROR_CHANNEL            ((uint8_t)32U)

/**
 * @brief   Maximum number of Group1 DSS ESM Error channels: As per the TRM
 * this is limited to 64
 */
#define DSS_ESM_MAX_GROUP1_ERROR_CHANNEL        ((uint8_t)64U)

/**
 * @brief   Maximum number of Group2 DSS ESM Error channels: As per the TRM
 * this is limited to 32
 */
#define DSS_ESM_MAX_GROUP2_ERROR_CHANNEL        ((uint8_t)32U)


/**************************************************************************
 ***************************** Local Structures ***************************
 **************************************************************************/

/**
 * @brief
 *  ESM Driver Interface MCB
 *
 * @details
 *  The structure is used to hold information pertinent to ESM Driver
 *  OSAL interface.
 *
 *  \ingroup OSAL_INTERNAL_STRUCTURES
 */
typedef struct OSAL_R4F_ESMDrv_MCB_t
{
    /**
     * @brief   Hook Function Table: This is indexed by the error channel
     * This table is used to handle Group1 (64) errors.
     */
    OSAL_HookFxn     group1HookTable[ESM_MAX_GROUP1_ERROR_CHANNEL];

    /**
     * @brief   Hook Function Table: This is indexed by the error channel
     * This table is used to handle Group2 (32) errors.
     */
    OSAL_HookFxn     group2HookTable[ESM_MAX_GROUP2_ERROR_CHANNEL];
}OSAL_R4F_ESMDrv_MCB;

/**************************************************************************
 **************************** Global Declarations *************************
 **************************************************************************/
#define MAX_ESM_NOTIFIER     4

typedef struct esmNotifierMapper
{
    uint32_t  errorNumber;
    uint32_t  notifierIdx;
}esmNotifierMapper_t;
    
esmNotifierMapper_t esmNotifMap[MAX_ESM_NOTIFIER] = {0};

/**************************************************************************
 ************************** Platform Definitions **************************
 **************************************************************************/

#if defined (SOC_XWR16XX)

/*************************************************************************
 * R4F XWR16xx:-
 * ---------------------------------------------------------------
 * | VIM Interrupt  | Description                                |
 * ---------------------------------------------------------------
 * | 0              | R4F VIM High Level Interrupt               |
 * | 20             | R4F VIM Low Level Interrupt                |
 * ---------------------------------------------------------------
 ************************************************************************/

/**
 * @brief   Platform Abstraction: Redefine the VIM High Priority Interrupt
 * definiton.  This will ensure that the ESM Driver Interface is platform independent.
 */
#define OSAL_ESM_VIM_HIGH_PRIORITY_INT       CSL_XWR16XX_R4F_VIM_ESM_HIGH_PRIORITY_INT

/**
 * @brief   Platform Abstraction: Redefine the VIM Low Priority Interrupt
 * definiton.  This will ensure that the ESM Driver Interface is platform independent.
 */
#define OSAL_ESM_VIM_LOW_PRIORITY_INT        CSL_XWR16XX_R4F_VIM_ESM_LOW_PRIORITY_INT

#elif defined (SOC_XWR18XX)

/*************************************************************************
 * R4F XWR18xx:-
 * ---------------------------------------------------------------
 * | VIM Interrupt  | Description                                |
 * ---------------------------------------------------------------
 * | 0              | R4F VIM High Level Interrupt               |
 * | 20             | R4F VIM Low Level Interrupt                |
 * ---------------------------------------------------------------
 ************************************************************************/

/**
 * @brief   Platform Abstraction: Redefine the VIM High Priority Interrupt
 * definiton.  This will ensure that the ESM Driver Interface is platform independent.
 */
#define OSAL_ESM_VIM_HIGH_PRIORITY_INT       CSL_XWR18XX_R4F_VIM_ESM_HIGH_PRIORITY_INT

/**
 * @brief   Platform Abstraction: Redefine the VIM Low Priority Interrupt
 * definiton.  This will ensure that the ESM Driver Interface is platform independent.
 */
#define OSAL_ESM_VIM_LOW_PRIORITY_INT        CSL_XWR18XX_R4F_VIM_ESM_LOW_PRIORITY_INT

#elif defined (SOC_XWR68XX)

/*************************************************************************
 * R4F XWR18xx:-
 * ---------------------------------------------------------------
 * | VIM Interrupt  | Description                                |
 * ---------------------------------------------------------------
 * | 0              | R4F VIM High Level Interrupt               |
 * | 20             | R4F VIM Low Level Interrupt                |
 * ---------------------------------------------------------------
 ************************************************************************/

/**
 * @brief   Platform Abstraction: Redefine the VIM High Priority Interrupt
 * definiton.  This will ensure that the ESM Driver Interface is platform independent.
 */
#define OSAL_ESM_VIM_HIGH_PRIORITY_INT       CSL_XWR68XX_R4F_VIM_ESM_HIGH_PRIORITY_INT

/**
 * @brief   Platform Abstraction: Redefine the VIM Low Priority Interrupt
 * definiton.  This will ensure that the ESM Driver Interface is platform independent.
 */
#define OSAL_ESM_VIM_LOW_PRIORITY_INT        CSL_XWR68XX_R4F_VIM_ESM_LOW_PRIORITY_INT

#endif

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/
/* For TI-RTOS, these handlers for low/high priority ESM interrupt, implemented in
 * ESM driver of mmWave SDK c:\ti\mmwave_sdk_03_05_00_01\packages\ti\drivers\esm.
 * ESM_lowpriority_IRQ, ESM_highpriority_FIQ
 */

/**
 *  @b Description
 *  @n
 *      In a working system the ESM handlers are owned by the
 *      application. However when a diagnostic is executed it
 *      would require a callback ("Hook") to be invoked by the
 *      application ESM handler. This is needed since the diagnostic
 *      will perform the necessary book keeping in the context
 *      of this function. Hooks are added during the start of
 *      the diagnostic execution.
 *
 *      The function is used to install a hook function for
 *      the ESM Group/Error number.
 *
 *  @param[in] ptrErrorChannel
 *      Pointer to the ESM error channel for which the notify function
 *      is to be installed.
 *  @param[in] fxn
 *      Error Hook Function which is invoked once the error is detected
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_ESMDrv_addHook (const CSL_ESM_ErrorChannel* ptrErrorChannel, OSAL_HookFxn fxn)
{
    /* For TI-RTOS, it uses mmWave-SDK ESM driver's register Notifier function */
    ESM_NotifyParams        notifyParams;
    int32_t                 errCode = 0;
    int32_t                 retVal = 0;
    uint8_t  i;

    retVal = 0;
    notifyParams.groupNumber = (uint32_t)ptrErrorChannel->group;
    notifyParams.errorNumber = (uint32_t)ptrErrorChannel->error;
    notifyParams.arg = NULL;
    notifyParams.notify = (ESM_CallBack)fxn;
    retVal = ESM_registerNotifier (gEsmHandle, &notifyParams, &errCode);
    
    for(i = 0; i < MAX_ESM_NOTIFIER; i++)
    {
        if(esmNotifMap[i].errorNumber == 0)
        {
            esmNotifMap[i].errorNumber = ptrErrorChannel->error;
            esmNotifMap[i].notifierIdx = retVal;
            break;
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the previous installed hook for
 *      the ESM Group/Error number
 *
 *  @param[in] ptrErrorChannel
 *      Pointer to the ESM error channel information for which
 *      the notify function is needed.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_ESMDrv_delHook (const CSL_ESM_ErrorChannel* ptrErrorChannel)
{
    /* For TI-RTOS, it uses mmWave-SDK ESM driver's De-register Notifier function */
    int32_t     errCode = 0;
    uint8_t     i;

    /* find the notifier index registered for ESM error num */
    for(i = 0; i < MAX_ESM_NOTIFIER; i++)
    {
        if(esmNotifMap[i].errorNumber == (uint32_t)ptrErrorChannel->error)
            break;        
    }
    
    ESM_deregisterNotifier (gEsmHandle, esmNotifMap[i].notifierIdx, &errCode);

    esmNotifMap[i].errorNumber = 0;
    esmNotifMap[i].notifierIdx = 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the ESM Driver Interface.
 *
 *      This is a reference only implementation which is invoked
 *      only from the Unit Tests.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 *  @note
 *    This should only be invoked after the OSAL Interrupt module
 *    has been initialized. The OSAL ESM Driver interface would
 *    register interrupts with the VIM module.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_ESMDrv_init (void)
{
    /* For TI-RTOS, it uses mmWave-SDK ESM driver's Init function */
    uint8_t bClearErrors = 0;
    /* use mmwave SDK ESM driver initialization here */
    gEsmHandle = ESM_init(bClearErrors);    
}

/************************************************************************************************************
 ************************************************************************************************************
 *  OSAL Functions to handle DSS ESM from R4F
 ************************************************************************************************************
 ************************************************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is used to get information about which enable DSS ESM interrupt
 *      triggered the MSS ESM.
 *
 *  @param[out] ptrDSSErrorChannel
 *      Populated with the error channel information for the pending DSS ESM interrupt
 *      if the return value of the function is non zero.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      0  - No pending interrupt
 *  @retval
 *      Any other value - Pending interrupt. ptrDSSErrorChannel will be filled with error information.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
uint8_t OSAL_R4F_DSSESMDrv_getInterrupt(CSL_ESM_ErrorChannel* ptrDSSErrorChannel)
{
    CSL_ESMRegs*    ptrDSSESMRegs;
    uint8_t         pendingError;

    /* Get the base address of the DSS ESM module: */
    ptrDSSESMRegs = CSL_DSS_ESM_getBaseAddress();

    /* Get the high level DSS pending error: */
    pendingError = CSL_ESM_getHighLevelPendingInterrupt(ptrDSSESMRegs, ptrDSSErrorChannel);

    /* Was there a high level DSS pending interrupt */
    if (pendingError == 0U)
    {
        /* No high level pending interrupt. Get the low level DSS pending error: */
        pendingError = CSL_ESM_getLowLevelPendingInterrupt(ptrDSSESMRegs, ptrDSSErrorChannel);
    }
    return pendingError;
}

/**
 *  @b Description
 *  @n
 *      The function is used to enable DSS ESM interrupt for the specified DSS ESM Group/Error number.
 *      Separately the MSS ESM for DSS Group1/Group2 error and interrupt handler needs to be enabled.
 *
 *  @param[in] ptrDSSErrorChannel
 *      Pointer to the DSS ESM error channel to be enabled.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_DSSESMDrv_enableInterrupt(const CSL_ESM_ErrorChannel* ptrDSSErrorChannel)
{
    CSL_ESMRegs*    ptrDSSESMRegs;
    CSL_DSSRegs*    ptrDSSRegs;

    /* Get the base address of the DSS ESM module: */
    ptrDSSESMRegs = CSL_DSS_ESM_getBaseAddress();

    /* Get the base address of DSS Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Is this a group 1 or group 2 interrupt? */
    if (ptrDSSErrorChannel->group == CSL_ESM_Group_1)
    {
        /* Group1 Interrupts: */
        if (ptrDSSErrorChannel->error < DSS_ESM_MAX_GROUP1_ERROR_CHANNEL)
        {
            /* Enable interrupt for Group1 Error: */
            CSL_ESM_enableErrorInterrupt (ptrDSSESMRegs, ptrDSSErrorChannel);
        }
    }
    else /* Group 2 DSS ESM error */
    {
        /* Group2 Interrupts: */
        if (ptrDSSErrorChannel->error < DSS_ESM_MAX_GROUP2_ERROR_CHANNEL)
        {
            /* Unmask the Group2 error */
            CSL_DSS_enableGroup2ESMError(ptrDSSRegs, ptrDSSErrorChannel->error);
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to disable DSS ESM interrupt for the specified DSS ESM Group/Error number.
 *      Separately the MSS ESM for DSS Group1/Group2 error and interrupt handler needs to be disabled.
 *
 *  @param[in] ptrDSSErrorChannel
 *      Pointer to the DSS ESM error channel to be disabled.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_DSSESMDrv_disableInterrupt(const CSL_ESM_ErrorChannel* ptrDSSErrorChannel)
{
    CSL_ESMRegs*    ptrDSSESMRegs;
    CSL_DSSRegs*    ptrDSSRegs;

    /* Get the base address of the DSS ESM module: */
    ptrDSSESMRegs = CSL_DSS_ESM_getBaseAddress();

    /* Get the base address of DSS Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Is this a group 1 or group 2 interrupt? */
    if (ptrDSSErrorChannel->group == CSL_ESM_Group_1)
    {
        /* Group1 Interrupts: */
        if (ptrDSSErrorChannel->error < DSS_ESM_MAX_GROUP1_ERROR_CHANNEL)
        {
            /* Disable interrupt for Group1 Error: */
            CSL_ESM_disableErrorInterrupt (ptrDSSESMRegs, ptrDSSErrorChannel);
        }
    }
    else /* Group 2 DSS ESM error */
    {
        /* Group2 Interrupts: */
        if (ptrDSSErrorChannel->error < DSS_ESM_MAX_GROUP2_ERROR_CHANNEL)
        {
            /* Mask the Group2 error */
            CSL_DSS_disableGroup2ESMError(ptrDSSRegs, ptrDSSErrorChannel->error);
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to clear DSS ESM interrupt for the specified DSS ESM Group/Error number.
 *      Separately the MSS ESM for DSS Group1/Group2 error needs to be cleared.
 *
 *  @param[in] ptrDSSErrorChannel
 *      Pointer to the DSS ESM error channel to be cleared.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_DSSESMDrv_clearInterrupt(const CSL_ESM_ErrorChannel* ptrDSSErrorChannel)
{
    CSL_ESMRegs*    ptrDSSESMRegs;

    /* Get the base address of the DSS ESM module: */
    ptrDSSESMRegs = CSL_DSS_ESM_getBaseAddress();

    /* Clear the pending error: */
    CSL_ESM_clearPendingError (ptrDSSESMRegs, ptrDSSErrorChannel);
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the DSS ESM Driver Interface.
 *
 *      This is a reference only implementation which is invoked
 *      only from the Unit Tests.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 *  @note
 *    This should only be invoked if MSS controls DSS ESM. This is needed only when
 *    MSS runs DSS diagnostics.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_DSSESMDrv_init (void)
{
    CSL_ESMRegs*            ptrDSSESMRegs;
    uint8_t                 index;
    CSL_ESM_ErrorChannel    errorChannel;

    /* Get the base address of the DSS ESM module: */
    ptrDSSESMRegs = CSL_DSS_ESM_getBaseAddress();

    /* Clear all the Group1 errors: */
    for (index = 0U; index < DSS_ESM_MAX_GROUP1_ERROR_CHANNEL; index++)
    {
        /* Populate the error channel: */
        errorChannel.group = CSL_ESM_Group_1;
        errorChannel.error = index;

        /* Clear the pending error: */
        CSL_ESM_clearPendingError (ptrDSSESMRegs, &errorChannel);

        /* Disable interrupts for Group1 Error: */
        CSL_ESM_disableErrorInterrupt (ptrDSSESMRegs, &errorChannel);

        /* Group1 Error are mapped to Low Priority: */
        CSL_ESM_setInterruptPriority (ptrDSSESMRegs, &errorChannel, 0U);
    }

    /* Clear all the Group2 errors: */
    for (index = 0U; index < DSS_ESM_MAX_GROUP2_ERROR_CHANNEL; index++)
    {
        /* Populate the error channel: */
        errorChannel.group = CSL_ESM_Group_2;
        errorChannel.error = index;

        /* Clear the pending error: */
        CSL_ESM_clearPendingError (ptrDSSESMRegs, &errorChannel);
    }
}
