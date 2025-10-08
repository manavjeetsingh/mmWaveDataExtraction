/*
 *   @file  osal_c674_esmdrv_nonos.c
 *
 *   @brief
 *          This file implements the C674 Non-OS OSAL ESM Driver Interface.
 *          File is mainly used for SDL Diagnostic Tests purpose.
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
#include <ti/csl/csl.h>
#include "osal/osal.h"

/**************************************************************************
 **************************** Local Declarations **************************
 **************************************************************************/

static void OSAL_C674_ESMDrv_defaultHandler (void);

/**************************************************************************
 **************************** Extern Declarations *************************
 **************************************************************************/

/* This function is used only by the Test code to initialize the OSAL R4F
 * ESM Driver module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_C674_ESMDrv_init (void);

/**************************************************************************
 **************************** Local Definitions ***************************
 **************************************************************************/

/**
 * @brief   Maximum number of Group1 ESM Error channels: As per the TRM
 * this is limited to 64
 */
#define ESM_MAX_GROUP1_ERROR_CHANNEL    (64U)

/**
 * @brief   Maximum number of Group2 ESM Error channels: As per the TRM
 * this is limited to 32
 */
#define ESM_MAX_GROUP2_ERROR_CHANNEL    (32U)

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
typedef struct OSAL_C674_ESMDrv_MCB_t
{
    /**
     * @brief   Notify Function Table: This is indexed by the error channel
     * This table is used to handle Group1 (64) errors.
     */
    OSAL_HookFxn       group1HookTable[ESM_MAX_GROUP1_ERROR_CHANNEL];

    /**
     * @brief   Notify Function Table: This is indexed by the error channel
     * This table is used to handle Group2 (32) errors.
     */
    OSAL_HookFxn       group2HookTable[ESM_MAX_GROUP2_ERROR_CHANNEL];
}OSAL_C674_ESMDrv_MCB;

/**************************************************************************
 **************************** Global Declarations *************************
 **************************************************************************/

/**
 * @brief   ESM Driver MCB:
 */
OSAL_C674_ESMDrv_MCB     gESMDrvMCB;

/**************************************************************************
 ************************** Platform Definitions **************************
 **************************************************************************/

#if defined (SOC_XWR16XX)

/*************************************************************************
 * For XWR16xx:-
 * -------------------------------------------------------------
 * | DSP Event Id | Description                                |
 * -------------------------------------------------------------
 * | 32           | DSS ESM Low Priority Interrupt             |
 * -------------------------------------------------------------
 ************************************************************************/

/**
 * @brief   Platform Abstraction: Redefine the ESM Event Identifier.
 * This will make the rest of the code platform independent.
 */
#define SDL_ESM_EVENT_ID                    CSL_XWR16XX_C674_ESM_LOW_PRIORITY_EVENT

#elif defined (SOC_XWR18XX)

/*************************************************************************
 * For XWR18xx:-
 * -------------------------------------------------------------
 * | DSP Event Id | Description                                |
 * -------------------------------------------------------------
 * | 32           | DSS ESM Low Priority Interrupt             |
 * -------------------------------------------------------------
 ************************************************************************/

/**
 * @brief   Platform Abstraction: Redefine the ESM Event Identifier.
 * This will make the rest of the code platform independent.
 */
#define SDL_ESM_EVENT_ID                    CSL_XWR18XX_C674_ESM_LOW_PRIORITY_EVENT
#elif defined (SOC_XWR68XX)

/*************************************************************************
 * For XWR68xx:-
 * -------------------------------------------------------------
 * | DSP Event Id | Description                                |
 * -------------------------------------------------------------
 * | 32           | DSS ESM Low Priority Interrupt             |
 * -------------------------------------------------------------
 ************************************************************************/

/**
 * @brief   Platform Abstraction: Redefine the ESM Event Identifier.
 * This will make the rest of the code platform independent.
 */
#define SDL_ESM_EVENT_ID                    CSL_XWR68XX_C674_ESM_LOW_PRIORITY_EVENT
#endif

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      This is the *dummy* default handler for all errors. The function
 *      will get stuck in a loop indefinitely. This was done to catch
 *      all unexpected ESM errors.
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not applicable
 */
static void OSAL_C674_ESMDrv_defaultHandler (void)
{
    volatile uint32_t deadloop = 1U;
    while (deadloop == 1U)
    {
    }
}

/**
 *  @b Description
 *  @n
 *      This is the ISR which is registered with the FIQ dispatcher and
 *      is invoked on the high priority interrupt.
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
static void OSAL_C674_ESMDrv_highPriorityISR (void)
{
    CSL_ESMRegs*            ptrESMRegs;
    CSL_ESM_ErrorChannel    errorChannel;
    uint8_t                 pendingError;
    extern volatile cregister uint32_t EFR;
    extern volatile cregister uint32_t ECR;
    volatile uint32_t efr;

    /* Get the base address of the ESM module: */
    ptrESMRegs = CSL_ESM_getBaseAddress ();

    /* Get the pending error interrupt: */
    pendingError = CSL_ESM_getHighLevelPendingInterrupt(ptrESMRegs, &errorChannel);
    while (pendingError != 0U)
    {
        /* Invoke the registered notify function: */
        if (errorChannel.group == CSL_ESM_Group_1)
        {
            gESMDrvMCB.group1HookTable[errorChannel.error]();
        }
        else
        {
            gESMDrvMCB.group2HookTable[errorChannel.error]();
        }

        /* Clear the pending ESM error: */
        CSL_ESM_clearPendingError (ptrESMRegs, &errorChannel);

        /* Get the next pending error interrupt: */
        pendingError = CSL_ESM_getHighLevelPendingInterrupt(ptrESMRegs, &errorChannel);
    }

    /* Clear NMI */
    efr = EFR;
    ECR = efr;
}

/**
 *  @b Description
 *  @n
 *      This is the ISR which is registered with to handle
 *      the ESM Driver interrupts on the C674
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
static void OSAL_C674_ESMDrv_ISR (void)
{
    CSL_ESMRegs*            ptrESMRegs;
    CSL_ESM_ErrorChannel    errorChannel;
    uint8_t                 pendingError;

    /* Get the base address of the ESM module: */
    ptrESMRegs = CSL_ESM_getBaseAddress ();

    /* Get the pending error interrupt: */
    pendingError = CSL_ESM_getLowLevelPendingInterrupt(ptrESMRegs, &errorChannel);
    while (pendingError != 0U)
    {
        /* Clear the pending ESM error: */
        CSL_ESM_clearPendingError (ptrESMRegs, &errorChannel);

        /* Invoke the registered notify function: */
        if (errorChannel.group == CSL_ESM_Group_1)
        {
            gESMDrvMCB.group1HookTable[errorChannel.error]();
        }
        else
        {
            /* There is no way to set Group2 errors to low priority and hence 
             * this part should never get exercised.
             */
            gESMDrvMCB.group2HookTable[errorChannel.error]();
        }

        /* Get the next pending error interrupt: */
        pendingError = CSL_ESM_getLowLevelPendingInterrupt(ptrESMRegs, &errorChannel);
    }
}

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
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_ESMDrv_addHook (const CSL_ESM_ErrorChannel* ptrErrorChannel, OSAL_HookFxn fxn)
{
    CSL_ESMRegs*    ptrESMRegs;
    CSL_DSSRegs*    ptrDSSRegs;

    /* Get the base address of the ESM module: */
    ptrESMRegs = CSL_ESM_getBaseAddress ();

    /* Get the base address of DSS Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Is this a group 1 or group 2 interrupt? */
    if (ptrErrorChannel->group == CSL_ESM_Group_1)
    {
        /* There is a way to set Group1 errors to high priority and hence
         * it is possible to get into this portion of code.
         */
        /* Group1 Interrupts: */
        if (ptrErrorChannel->error < ESM_MAX_GROUP1_ERROR_CHANNEL)
        {
            /* Setup the notify function: */
            gESMDrvMCB.group1HookTable[ptrErrorChannel->error] = fxn;

            /* Enable interrupts for Group1 Error: */
            CSL_ESM_enableErrorInterrupt (ptrESMRegs, ptrErrorChannel);
        }
    }
    else if (ptrErrorChannel->group == CSL_ESM_Group_2)
    {
        /* Group2 Interrupts: */
        if (ptrErrorChannel->error < ESM_MAX_GROUP2_ERROR_CHANNEL)
        {
            /* Setup the notify function: */
            gESMDrvMCB.group2HookTable[ptrErrorChannel->error] = fxn;

            /* Enable the group2 error */
            CSL_DSS_enableGroup2ESMError(ptrDSSRegs, ptrErrorChannel->error);
        }
    }
    else
    {
        /* Group 3 Interrupts do not generate interrupts so a function
         * handler cannot be installed. */
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
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_ESMDrv_delHook (const CSL_ESM_ErrorChannel* ptrErrorChannel)
{
    CSL_ESMRegs*    ptrESMRegs;
    CSL_DSSRegs*    ptrDSSRegs;

    /* Get the base address of the ESM module: */
    ptrESMRegs = CSL_ESM_getBaseAddress ();

    /* Get the base address of DSS Module */
    ptrDSSRegs = CSL_DSS_getBaseAddress ();

    /* Is this a group 1 or group 2 interrupt? */
    if (ptrErrorChannel->group == CSL_ESM_Group_1)
    {
        /* Group1 Interrupts: */
        if (ptrErrorChannel->error < ESM_MAX_GROUP1_ERROR_CHANNEL)
        {
            /* Delete the hook: Add back the default handler */
            gESMDrvMCB.group1HookTable[ptrErrorChannel->error] = OSAL_C674_ESMDrv_defaultHandler;

            /* Disable interrupts for Group1 Error: */
            CSL_ESM_disableErrorInterrupt (ptrESMRegs, ptrErrorChannel);
        }
    }
    else if (ptrErrorChannel->group == CSL_ESM_Group_2)
    {
        /* Group2 Interrupts: */
        if (ptrErrorChannel->error < ESM_MAX_GROUP2_ERROR_CHANNEL)
        {
            /* Disable the group2 error */
            CSL_DSS_disableGroup2ESMError(ptrDSSRegs, ptrErrorChannel->error);

            /* Delete the hook: Add back the default handler */
            gESMDrvMCB.group2HookTable[ptrErrorChannel->error] = OSAL_C674_ESMDrv_defaultHandler;
        }
    }
    else
    {
        /* Group3 Interrupts: These do not generate interrupts so there
         * are no handlers installed for this group. */
    }
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
 *      The function is used to initialize the ESM Driver Interface.
 *      The ESM Driver interface should only be invoked once the
 *      Interrupt module has been initialized since this will
 *      register error ISR
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_ESMDrv_init (void)
{
    CSL_ESMRegs*            ptrESMRegs;
    uint8_t                 index;
    CSL_ESM_ErrorChannel    errorChannel;

    /* Initialize the ESM Driver MCB: */
    (void)memset ((void *)&gESMDrvMCB, 0, sizeof(OSAL_C674_ESMDrv_MCB));

    /* Get the base address of the ESM module: */
    ptrESMRegs = CSL_ESM_getBaseAddress ();

    /* Clear all the Group1 errors: */
    for (index = 0U; index < ESM_MAX_GROUP1_ERROR_CHANNEL; index++)
    {
        /* Populate the error channel: */
        errorChannel.group = CSL_ESM_Group_1;
        errorChannel.error = index;

        /* Clear the pending error: */
        CSL_ESM_clearPendingError (ptrESMRegs, &errorChannel);

        /* Setup the default handler for Group1 Errors: */
        gESMDrvMCB.group1HookTable[errorChannel.error] = OSAL_C674_ESMDrv_defaultHandler;

        /* Disable interrupts for Group1 Error: */
        CSL_ESM_disableErrorInterrupt (ptrESMRegs, &errorChannel);

        /* Group1 Error are mapped to Low Priority: */
        CSL_ESM_setInterruptPriority (ptrESMRegs, &errorChannel, 0U);
    }

    /* Clear all the Group2 errors: */
    for (index = 0U; index < ESM_MAX_GROUP2_ERROR_CHANNEL; index++)
    {
        /* Populate the error channel: */
        errorChannel.group = CSL_ESM_Group_2;
        errorChannel.error = index;

        /* Clear the pending error: */
        CSL_ESM_clearPendingError (ptrESMRegs, &errorChannel);

        /* Setup the default handler for Group2 Errors: */
        gESMDrvMCB.group2HookTable[errorChannel.error] = OSAL_C674_ESMDrv_defaultHandler;
    }

    /* Setup the ISR for ESM Processing: */
    OSAL_C674_Interrupt_addHook (SDL_ESM_EVENT_ID, OSAL_C674_ESMDrv_ISR);

    /* Setup the NMI for ESM group2 Processing: */
    OSAL_C674_NMI_addHook (OSAL_C674_ESMDrv_highPriorityISR);

    return;
}

