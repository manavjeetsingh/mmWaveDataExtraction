/*
 *   @file  osal_r4f_interrupt.c
 *
 *   @brief
 *          This file implements the R4F TI-RTOS Interrupt Management.
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
#include <stdint.h>

/* mmWave SDL Include Files */
#include "osal/osal.h"
#include "osal/hwiP.h"
#include <ti/csl/csl.h>
#include <osal/mmwave_sdk_port.h>
#include <ti/sysbios/knl/Task.h>


#define MAX_HWI_HANDLE     4
typedef struct hwiHandleMap_t
{
    HwiP_Handle hwiHandle;
    uint8_t     irq;
}hwiHandleMap;


hwiHandleMap gHwiHandleMap[MAX_HWI_HANDLE] = {0};

/**************************************************************************
 **************************** Local Declarations **************************
 **************************************************************************/

static void OSAL_R4F_Interrupt_defaultHandler (void);
#if defined (DYNAMIC_ANALYSIS_ENABLED)
static void OSAL_R4F_Interrupt_emptyHandler (void);
#endif

/**************************************************************************
 **************************** Extern Declarations *************************
 **************************************************************************/

/* This function is used only by the Test code to initialize the OSAL R4F
 * Interrupt module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_R4F_InterruptInit (void);

/**************************************************************************
 ***************************** Local Structures ***************************
 **************************************************************************/

/**
 * @brief
 *  R4F Interrupt Vector Table
 *
 * @details
 *  The structure is used to hold information of the registered
 *  interrupt handlers which are invoked on the reception of the
 *  R4F interrupt.
 *
 *  \ingroup OSAL_INTERNAL_STRUCTURES
 */
typedef struct OSAL_R4F_Interrupt_VectorTable_t
{
    /**
     * @brief   Undefined Instruction Handler
     */
    OSAL_HookFxn    undefInstrHandler;

    /**
     * @brief   Data Abort Handler
     */
    OSAL_HookFxn    dataAbortHandler;

    /**
     * @brief   SVC Handler
     */
    OSAL_HookFxn    svcHandler;

    /**
     * @brief   Prefetch Abort Handler
     */
    OSAL_HookFxn    prefetchAbortHandler;
}OSAL_R4F_Interrupt_VectorTable;

/**************************************************************************
 **************************** Global Declarations *************************
 **************************************************************************/

/**
 * @brief   R4F Interrupt Vector Table:
 */
OSAL_R4F_Interrupt_VectorTable  gOSAL_R4F_InterruptVectorTable =
{
    OSAL_R4F_Interrupt_defaultHandler,  /* Undefined Instruction Handler:   */
    OSAL_R4F_Interrupt_defaultHandler,  /* Data Abort Handler:              */
    OSAL_R4F_Interrupt_defaultHandler,  /* SVC Handler:                     */
#if defined (DYNAMIC_ANALYSIS_ENABLED)
    OSAL_R4F_Interrupt_emptyHandler    /* Prefetch Abort Handler:          */
#else
    OSAL_R4F_Interrupt_defaultHandler   /* Prefetch Abort Handler:          */
#endif
};

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/
#if 0
/* For TI-RTOS (SysBios), user-defined exception handler is attached with OS
 * so this function gets called instead of SysBios Internal exception handler.
 * This handler function is attached with OS by mmw_mss.cfg.
 *
 * Based on different Diagnostic test, it connects its own handler function
 * before generating any exception (like CPU data abort) during diagnostic test.
 */
void MMW_EXCEPTION_HANDLER(Exception_ExcContext *excContext)
{
    /* check type of exception and call relevant handler based on that */
    if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_PreAbort)
    {
        gOSAL_R4F_InterruptVectorTable.prefetchAbortHandler();
    }
    else if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_DataAbort)
    {
        gOSAL_R4F_InterruptVectorTable.dataAbortHandler();
    }
    else if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_UndefInst)
    {
        gOSAL_R4F_InterruptVectorTable.undefInstrHandler ();
    }
    else if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_Supervisor)
    {
        /* ti_sysbios_family_arm_exc_Exception_Type_Supervisor */
        gOSAL_R4F_InterruptVectorTable.svcHandler();
    }
    else
    {
        excContext->type = (ti_sysbios_family_arm_exc_Exception_Type)0;
    }

}
#endif
/**
 *  @b Description
 *  @n
 *      Default Interrupt/CPU exception handler which is installed
 *      for all the interrupts & CPU exceptions. The function will
 *      get stuck in a loop indefinitely. This was done to catch
 *      all unexpected interrupts/CPU exceptions
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_R4F_Interrupt_defaultHandler (void)
{
    volatile uint32_t deadloop = 1U;
    while (deadloop == 1U)
    {
    }
}

/* For TI-RTOS it uses sysBios internal interrupt handler, where
 * Hooks are attached to the application to get any kind of fault/error caused
 * by Diagnostic Test. */

#if defined (DYNAMIC_ANALYSIS_ENABLED)
/**
 *  @b Description
 *  @n
 *      Empty Interrupt handler. This is used to ignore pre-fetch
 *      exception when STC test is run in CCS environment. In non-CCS
 *      environment this is most probably not needed.
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_R4F_Interrupt_emptyHandler (void)
{
    /* Do nothing */
}
#endif

/**
 *  @b Description
 *  @n
 *      In a working system the exceptions are owned by the application
 *      However when a diagnostic is executed it would require a
 *      callback ("Hook") to be invoked by the application exception
 *      handler. This is needed since the diagnostic will perform
 *      the necessary book keeping in the context of this function.
 *      Hooks are added during the start of the diagnostic execution.
 *
 *      The function is used to install a hook to the R4F CPU
 *      exception handler. The hook function will be triggered
 *      on the CPU exception.
 *
 *  @param[in] type
 *      Exception Type
 *  @param[in] fxn
 *      Exception Hook to be installed
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_Interrupt_addExceptionHook (OSAL_R4F_Exception type, OSAL_HookFxn fxn)
{
    switch (type)
    {
        case OSAL_R4F_Exception_PREFETCH_ABORT:
        {
            gOSAL_R4F_InterruptVectorTable.prefetchAbortHandler = fxn;
            break;
        }
        case OSAL_R4F_Exception_DATA_ABORT:
        {
            gOSAL_R4F_InterruptVectorTable.dataAbortHandler = fxn;
            break;
        }
        case OSAL_R4F_Exception_UNDEFINED_INSTR:
        {
            gOSAL_R4F_InterruptVectorTable.undefInstrHandler = fxn;
            break;
        }
        case OSAL_R4F_Exception_SVC:
        {
            gOSAL_R4F_InterruptVectorTable.svcHandler = fxn;
            break;
        }
        default:
        {
            /* This is a catch-all condition. Control should never come here because this
             * violates the enumeration range. */
            break;
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the hook for the R4F CPU
 *      exception handler.
 *
 *  @param[in] type
 *      Exception Type for which the hook is to be deleted
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_Interrupt_delExceptionHook (OSAL_R4F_Exception type)
{
    switch (type)
    {
        case OSAL_R4F_Exception_PREFETCH_ABORT:
        {
            gOSAL_R4F_InterruptVectorTable.prefetchAbortHandler = OSAL_R4F_Interrupt_defaultHandler;
            break;
        }
        case OSAL_R4F_Exception_DATA_ABORT:
        {
            gOSAL_R4F_InterruptVectorTable.dataAbortHandler = OSAL_R4F_Interrupt_defaultHandler;
            break;
        }
        case OSAL_R4F_Exception_UNDEFINED_INSTR:
        {
            gOSAL_R4F_InterruptVectorTable.undefInstrHandler = OSAL_R4F_Interrupt_defaultHandler;
            break;
        }
        case OSAL_R4F_Exception_SVC:
        {
            gOSAL_R4F_InterruptVectorTable.svcHandler = OSAL_R4F_Interrupt_defaultHandler;
            break;
        }
        default:
        {
            /* This is a catch-all condition. Control should never come here because this
             * violates the enumeration range. */
            break;
        }
    }
}

/**
 *  @b Description
 *  @n
 *      In a working system the ISR is owned by the application
 *      However when a diagnostic is executed it would require a
 *      callback ("Hook") to be invoked by the application ISR. This
 *      is needed since the diagnostic will perform the necessary
 *      book keeping in the context of this function. Hooks are
 *      added during the start of the diagnostic execution.
 *
 *      The function is used to add the specific hook function
 *      which is to be invoked on the specified interrupt.
 *
 *  @param[in] irq
 *      Interrupt Request Number
 *  @param[in] fiqType                  \n
 *      Set the flag as follows:-       \n
 *          0 - IRQ                     \n
 *          1 - FIQ                     \n
 *  @param[in] hookFxn
 *      Hook Function to be invoked
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_Interrupt_addHook (uint8_t irq, uint8_t fiqType, OSAL_HookFxn hookFxn)
{
    /* For TI-RTOS it uses mmwave SDK Hwi OSAL HwiP layer */
    HwiP_Params     hwiParams;
    uint8_t idx;

    /* find the empty Hwi Handle map */
    for(idx=0; idx<MAX_HWI_HANDLE; idx++)
    {
        if((gHwiHandleMap[idx].hwiHandle == NULL) && (gHwiHandleMap[idx].irq == 0))
            break;
    }

    HwiP_Params_init(&hwiParams);
    hwiParams.name = "SDL_IRQ_HOOK";
    /* HwiP_Type_IRQ or HwiP_Type_FIQ; */
    hwiParams.type = (HwiP_Type)fiqType;
    gHwiHandleMap[idx].irq = irq;
    gHwiHandleMap[idx].hwiHandle = HwiP_create(irq, (HwiP_Fxn)hookFxn, &hwiParams);
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the specific hook function
 *      which was to be invoked on the specified interrupt.
 *
 *  @param[in] irq
 *      Interrupt Request Number
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_Interrupt_delHook (uint8_t irq)
{
    /* For TI-RTOS it uses mmwave SDK Hwi OSAL HwiP layer */
    uint8_t idx;

    /* find the empty Hwi Handle map */
    for(idx=0; idx < MAX_HWI_HANDLE; idx++)
    {
        if(gHwiHandleMap[idx].irq == irq)
            break;
    }

    HwiP_delete(((HwiP_Handle)gHwiHandleMap[idx].hwiHandle));

    gHwiHandleMap[idx].hwiHandle = NULL;
    gHwiHandleMap[idx].irq       = 0;
}

/**
 *  @b Description
 *  @n
 *      The function has been provided as a utility function
 *      to initialize the VIM. Interrupt handling can only
 *      be done after this function has been invoked. This
 *      function is only invoked by the unit tests to verify
 *      diagnostic functionality. System executing with an
 *      embedded operating system would have this functionality
 *      as a part of the operating system startup procedure.
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_InterruptInit (void)
{
    /* For TI-RTOS it uses mmwave SDK Hwi OSAL HwiP clear interrupt */
    int index;
    for ( index = 0U; index <= 126U; index++)
    {
        HwiP_clearInterrupt(index);
    }
}
